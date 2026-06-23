#pragma once

#include "db/mysql_pool.h"
#include "session.h"
#include "session_storage.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct ChatMessage
{
  std::string role;
  std::string content;
  std::int64_t created_at_epoch{0};
};

struct SessionConfig
{
  std::string user_file;
  std::string db_config_file;
  std::int64_t ttl_seconds{3600};
};

class MysqlSessionStorage : public SessionStorage
{
public:
  explicit MysqlSessionStorage(std::shared_ptr<MysqlPool> pool);

  bool save(const SessionRecord &session) override;
  std::optional<SessionRecord> load(const std::string &session_id) override;
  bool remove(const std::string &session_id) override;
  void cleanup_expired(std::int64_t now_epoch) override;

private:
  bool ensure_table();
  std::string escape(MYSQL *conn, const std::string &input);

  std::shared_ptr<MysqlPool> pool_;
  bool table_ready_{false};
};

class SessionManager
{
public:
  static SessionManager &instance();

  bool ensure_initialized(const SessionConfig &config);
  std::optional<std::string> authenticate(const std::string &username,
                                          const std::string &password);
  bool register_user(const std::string &username,
                     const std::string &password,
                     std::string &error_message);
  bool append_chat_message(const std::string &session_id,
                           const std::string &user_id,
                           const std::string &role,
                           const std::string &content,
                           std::string &error_message);
  std::vector<ChatMessage> list_chat_messages(const std::string &session_id,
                                              std::size_t limit,
                                              std::string &error_message);
  std::vector<ChatMessage> list_user_chat_messages(const std::string &user_id,
                                                   std::size_t limit,
                                                   std::string &error_message);
  bool clear_user_chat_messages(const std::string &user_id,
                                std::string &error_message);
  std::string create_session(const std::string &user_id);
  std::optional<SessionRecord> get_session(const std::string &session_id);
  bool delete_session(const std::string &session_id);

private:
  SessionManager() = default;

  std::string generate_session_id();
  std::int64_t to_epoch(std::chrono::system_clock::time_point tp) const;
  bool ensure_chat_table();

  bool load_users(const std::string &path);
  bool save_users(const std::string &path) const;
  bool initialized_{false};

  SessionConfig config_{};
  std::shared_ptr<MysqlPool> pool_;
  std::unique_ptr<SessionStorage> storage_;
  std::unordered_map<std::string, std::string> users_;
  bool chat_table_ready_{false};

  std::mutex mutex_;
  std::unordered_map<std::string, SessionRecord> cache_;
};
