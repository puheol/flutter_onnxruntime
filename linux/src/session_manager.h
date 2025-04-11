#ifndef SESSION_MANAGER_H
#define SESSION_MANAGER_H

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Session information structure
struct SessionInfo {
  void *session;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

// Session Manager Class
class SessionManager {
public:
  SessionManager();
  ~SessionManager();

  // Create a new session from a model file path
  std::string createSession(const char *model_path, void *options);

  // Get a session by ID
  void *getSession(const std::string &session_id);

  // Close and remove a session
  bool closeSession(const std::string &session_id);

  // Get session info
  SessionInfo *getSessionInfo(const std::string &session_id);

  // Check if a session exists
  bool hasSession(const std::string &session_id);

  // Get input names for a session
  std::vector<std::string> getInputNames(const std::string &session_id);

  // Get output names for a session
  std::vector<std::string> getOutputNames(const std::string &session_id);

private:
  // Generate a unique session ID
  std::string generateSessionId();

  // Map of session IDs to session info
  std::map<std::string, SessionInfo> sessions_;

  // Counter for generating unique session IDs
  int next_session_id_;

  // Mutex for thread safety
  std::mutex mutex_;
};

#endif // SESSION_MANAGER_H