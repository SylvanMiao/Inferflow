#include "session/session_manager.h"

#include "router/program_state_test.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <algorithm>
#include <cstring>

SessionManager &SessionManager::instance()
{
  static SessionManager manager;
  return manager;
}

bool SessionManager::ensure_initialized(const SessionConfig &config)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_)
  {
    return true;
  }

  config_ = config;
  if (!load_users(config_.user_file))
  {
    return false;
  }

  std::string db_text;
  if (!program_state_test::load_text_file(config_.db_config_file, db_text))
  {
    return false;
  }

  Json::Value root;
  Json::Reader reader;
  if (!reader.parse(db_text, root))
  {
    return false;
  }

  MysqlConfig mysql_config;
  mysql_config.host = root.get("host", "127.0.0.1").asString();
  mysql_config.port = root.get("port", 3306).asUInt();
  mysql_config.user = root.get("user", "root").asString();
  mysql_config.password = root.get("password", "").asString();
  mysql_config.database = root.get("database", "notix").asString();
  mysql_config.pool_size = root.get("pool_size", 8).asUInt();
  mysql_config.connect_timeout_sec = root.get("connect_timeout_sec", 5).asUInt();

  pool_ = std::make_shared<MysqlPool>(mysql_config);
  storage_ = std::make_unique<MysqlSessionStorage>(pool_);

  initialized_ = true;
  return true;
}

bool SessionManager::load_users(const std::string &path)
{
  std::string content;
  if (!program_state_test::load_text_file(path, content))
  {
    return false;
  }

  Json::Value root;
  Json::Reader reader;
  if (!reader.parse(content, root))
  {
    return false;
  }

  users_.clear();
  const auto users = root["users"];
  if (!users.isArray())
  {
    return false;
  }

  for (const auto &entry : users)
  {
    if (!entry.isMember("username") || !entry.isMember("password"))
    {
      continue;
    }
    users_[entry["username"].asString()] = entry["password"].asString();
  }

  return true;
}

bool SessionManager::save_users(const std::string &path) const
{
  Json::Value root;
  root["users"] = Json::arrayValue;

  for (const auto &entry : users_)
  {
    Json::Value user;
    user["username"] = entry.first;
    user["password"] = entry.second;
    root["users"].append(user);
  }

  const auto serialized = root.toStyledString();
  auto write_one = [&serialized](const std::string &target_path)
  {
    std::ofstream out(target_path, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
      return false;
    }
    out << serialized;
    return out.good();
  };

  const bool primary_ok = write_one(path);

  // 兼容从 build 目录启动服务的场景，同时回写到源码 static 目录。
  if (path.rfind("static/", 0) == 0)
  {
    write_one("../" + path);
  }

  return primary_ok;
}

std::optional<std::string> SessionManager::authenticate(const std::string &username,
                                                        const std::string &password)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = users_.find(username);
  if (it == users_.end())
  {
    return std::nullopt;
  }

  if (it->second != password)
  {
    return std::nullopt;
  }

  return username;
}

bool SessionManager::register_user(const std::string &username,
                                   const std::string &password,
                                   std::string &error_message)
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (username.empty())
  {
    error_message = "username must not be empty";
    return false;
  }

  if (password.size() < 6)
  {
    error_message = "password must be at least 6 characters";
    return false;
  }

  if (users_.find(username) != users_.end())
  {
    error_message = "username already exists";
    return false;
  }

  users_[username] = password;
  if (!save_users(config_.user_file))
  {
    users_.erase(username);
    error_message = "persist user failed";
    return false;
  }

  return true;
}

std::string SessionManager::generate_session_id()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 15);

  std::ostringstream ss;
  for (int i = 0; i < 32; ++i)
  {
    ss << std::hex << dist(gen);
  }
  return ss.str();
}

std::int64_t SessionManager::to_epoch(std::chrono::system_clock::time_point tp) const
{
  return std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch()).count();
}

bool SessionManager::ensure_chat_table()
{
  if (chat_table_ready_)
  {
    return true;
  }

  if (!pool_)
  {
    return false;
  }

  auto guard = pool_->acquire();
  MYSQL *conn = guard.get();
  const char *sql =
      "CREATE TABLE IF NOT EXISTS chat_messages ("
      "id BIGINT AUTO_INCREMENT PRIMARY KEY,"
      "session_id VARCHAR(64) NOT NULL,"
      "user_id VARCHAR(64) NOT NULL,"
      "role VARCHAR(16) NOT NULL,"
      "content TEXT NOT NULL,"
      "created_at BIGINT NOT NULL,"
      "INDEX idx_session_id_id (session_id, id)"
      ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;";

  if (mysql_real_query(conn, sql, std::strlen(sql)) != 0)
  {
    return false;
  }

  chat_table_ready_ = true;
  return true;
}

bool SessionManager::append_chat_message(const std::string &session_id,
                                         const std::string &user_id,
                                         const std::string &role,
                                         const std::string &content,
                                         std::string &error_message)
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (session_id.empty() || user_id.empty() || role.empty() || content.empty())
  {
    error_message = "invalid chat payload";
    return false;
  }

  if (!ensure_chat_table())
  {
    error_message = "prepare chat table failed";
    return false;
  }

  auto guard = pool_->acquire();
  MYSQL *conn = guard.get();

  auto escape = [conn](const std::string &input)
  {
    std::string escaped;
    escaped.resize(input.size() * 2 + 1);
    unsigned long len = mysql_real_escape_string(conn, &escaped[0], input.c_str(), input.size());
    escaped.resize(len);
    return escaped;
  };

  const auto sid = escape(session_id);
  const auto uid = escape(user_id);
  const auto safe_role = escape(role);
  const auto safe_content = escape(content);
  const auto now = to_epoch(std::chrono::system_clock::now());

  std::ostringstream ss;
  ss << "INSERT INTO chat_messages (session_id, user_id, role, content, created_at) VALUES ('"
     << sid << "','" << uid << "','" << safe_role << "','" << safe_content << "'," << now << ");";

  const auto sql = ss.str();
  if (mysql_real_query(conn, sql.c_str(), sql.size()) != 0)
  {
    error_message = "insert chat message failed";
    return false;
  }

  return true;
}

std::vector<ChatMessage> SessionManager::list_chat_messages(const std::string &session_id,
                                                            std::size_t limit,
                                                            std::string &error_message)
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<ChatMessage> result;

  if (session_id.empty())
  {
    error_message = "session id is empty";
    return result;
  }

  if (limit == 0)
  {
    return result;
  }

  if (!ensure_chat_table())
  {
    error_message = "prepare chat table failed";
    return result;
  }

  auto guard = pool_->acquire();
  MYSQL *conn = guard.get();

  std::string sid;
  sid.resize(session_id.size() * 2 + 1);
  const auto sid_len = mysql_real_escape_string(conn, &sid[0], session_id.c_str(), session_id.size());
  sid.resize(sid_len);

  std::ostringstream ss;
  ss << "SELECT role, content, created_at FROM chat_messages WHERE session_id='"
     << sid << "' ORDER BY id DESC LIMIT " << limit << ";";
  const auto sql = ss.str();

  if (mysql_real_query(conn, sql.c_str(), sql.size()) != 0)
  {
    error_message = "query chat history failed";
    return result;
  }

  MYSQL_RES *res = mysql_store_result(conn);
  if (!res)
  {
    return result;
  }

  MYSQL_ROW row;
  while ((row = mysql_fetch_row(res)) != nullptr)
  {
    ChatMessage item;
    item.role = row[0] ? row[0] : "";
    item.content = row[1] ? row[1] : "";
    item.created_at_epoch = row[2] ? std::stoll(row[2]) : 0;
    result.push_back(std::move(item));
  }

  mysql_free_result(res);
  std::reverse(result.begin(), result.end());
  return result;
}

std::vector<ChatMessage> SessionManager::list_user_chat_messages(const std::string &user_id,
                                                                 std::size_t limit,
                                                                 std::string &error_message)
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<ChatMessage> result;

  if (user_id.empty())
  {
    error_message = "user id is empty";
    return result;
  }

  if (limit == 0)
  {
    return result;
  }

  if (!ensure_chat_table())
  {
    error_message = "prepare chat table failed";
    return result;
  }

  auto guard = pool_->acquire();
  MYSQL *conn = guard.get();

  std::string uid;
  uid.resize(user_id.size() * 2 + 1);
  const auto uid_len = mysql_real_escape_string(conn, &uid[0], user_id.c_str(), user_id.size());
  uid.resize(uid_len);

  std::ostringstream ss;
  ss << "SELECT role, content, created_at FROM chat_messages WHERE user_id='"
     << uid << "' ORDER BY id DESC LIMIT " << limit << ";";
  const auto sql = ss.str();

  if (mysql_real_query(conn, sql.c_str(), sql.size()) != 0)
  {
    error_message = "query user chat history failed";
    return result;
  }

  MYSQL_RES *res = mysql_store_result(conn);
  if (!res)
  {
    return result;
  }

  MYSQL_ROW row;
  while ((row = mysql_fetch_row(res)) != nullptr)
  {
    ChatMessage item;
    item.role = row[0] ? row[0] : "";
    item.content = row[1] ? row[1] : "";
    item.created_at_epoch = row[2] ? std::stoll(row[2]) : 0;
    result.push_back(std::move(item));
  }

  mysql_free_result(res);
  std::reverse(result.begin(), result.end());
  return result;
}

bool SessionManager::clear_user_chat_messages(const std::string &user_id,
                                              std::string &error_message)
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (user_id.empty())
  {
    error_message = "user id is empty";
    return false;
  }

  if (!ensure_chat_table())
  {
    error_message = "prepare chat table failed";
    return false;
  }

  auto guard = pool_->acquire();
  MYSQL *conn = guard.get();

  std::string uid;
  uid.resize(user_id.size() * 2 + 1);
  const auto uid_len = mysql_real_escape_string(conn, &uid[0], user_id.c_str(), user_id.size());
  uid.resize(uid_len);

  std::ostringstream ss;
  ss << "DELETE FROM chat_messages WHERE user_id='" << uid << "';";
  const auto sql = ss.str();

  if (mysql_real_query(conn, sql.c_str(), sql.size()) != 0)
  {
    error_message = "clear chat history failed";
    return false;
  }

  return true;
}

std::string SessionManager::create_session(const std::string &user_id)
{
  const auto now = std::chrono::system_clock::now();
  SessionRecord record;
  record.session_id = generate_session_id();
  record.user_id = user_id;
  record.data_json = "{}";
  record.last_active_epoch = to_epoch(now);
  record.expires_at_epoch = to_epoch(now + std::chrono::seconds(config_.ttl_seconds));

  {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_[record.session_id] = record;
  }

  if (storage_)
  {
    storage_->save(record);
  }

  return record.session_id;
}

std::optional<SessionRecord> SessionManager::get_session(const std::string &session_id)
{
  const auto now_epoch = to_epoch(std::chrono::system_clock::now());

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(session_id);
    if (it != cache_.end())
    {
      if (it->second.expires_at_epoch < now_epoch)
      {
        cache_.erase(it);
        if (storage_)
        {
          storage_->remove(session_id);
        }
        return std::nullopt;
      }
      it->second.last_active_epoch = now_epoch;
      it->second.expires_at_epoch = now_epoch + config_.ttl_seconds;
      if (storage_)
      {
        storage_->save(it->second);
      }
      return it->second;
    }
  }

  if (storage_)
  {
    auto record = storage_->load(session_id);
    if (!record)
    {
      return std::nullopt;
    }
    if (record->expires_at_epoch < now_epoch)
    {
      storage_->remove(session_id);
      return std::nullopt;
    }

    record->last_active_epoch = now_epoch;
    record->expires_at_epoch = now_epoch + config_.ttl_seconds;
    storage_->save(*record);

    std::lock_guard<std::mutex> lock(mutex_);
    cache_[record->session_id] = *record;
    return record;
  }

  return std::nullopt;
}

bool SessionManager::delete_session(const std::string &session_id)
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.erase(session_id);
  }

  if (storage_)
  {
    return storage_->remove(session_id);
  }

  return true;
}
