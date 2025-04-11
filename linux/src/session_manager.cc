#include "session_manager.h"
#include <iostream>

SessionManager::SessionManager() : next_session_id_(1) {
  // No initialization needed for dummy implementation
}

SessionManager::~SessionManager() {
  // Clear all sessions
  std::lock_guard<std::mutex> lock(mutex_);
  sessions_.clear();
}

std::string SessionManager::createSession(const char *model_path, void *options) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Generate a session ID
  std::string session_id = generateSessionId();

  // Create a new session with dummy data
  SessionInfo session_info;
  session_info.session = nullptr; // Dummy session pointer

  // Add some dummy input names
  session_info.input_names.push_back("input1");
  session_info.input_names.push_back("input2");

  // Add some dummy output names
  session_info.output_names.push_back("output1");

  // Store the session info
  sessions_[session_id] = std::move(session_info);

  return session_id;
}

void *SessionManager::getSession(const std::string &session_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (sessions_.find(session_id) != sessions_.end()) {
    return sessions_[session_id].session;
  }

  return nullptr;
}

bool SessionManager::closeSession(const std::string &session_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = sessions_.find(session_id);
  if (it != sessions_.end()) {
    sessions_.erase(it);
    return true;
  }

  return false;
}

SessionInfo *SessionManager::getSessionInfo(const std::string &session_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = sessions_.find(session_id);
  if (it != sessions_.end()) {
    return &(it->second);
  }

  return nullptr;
}

bool SessionManager::hasSession(const std::string &session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return sessions_.find(session_id) != sessions_.end();
}

std::vector<std::string> SessionManager::getInputNames(const std::string &session_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = sessions_.find(session_id);
  if (it != sessions_.end()) {
    return it->second.input_names;
  }

  return {};
}

std::vector<std::string> SessionManager::getOutputNames(const std::string &session_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = sessions_.find(session_id);
  if (it != sessions_.end()) {
    return it->second.output_names;
  }

  return {};
}

std::string SessionManager::generateSessionId() { return "session_" + std::to_string(next_session_id_++); }