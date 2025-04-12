#include "session_manager.h"
#include <iostream>

SessionManager::SessionManager() : next_session_id_(1), env_(ORT_LOGGING_LEVEL_WARNING, "FlutterOnnxRuntime") {
  // Initialize ONNX Runtime environment in constructor
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

  try {
    // Create session options
    Ort::SessionOptions session_options;

    // If options are provided, use them
    if (options != nullptr) {
      // Cast the void* to the correct type
      Ort::SessionOptions *provided_options = static_cast<Ort::SessionOptions *>(options);

      // Directly use the provided options
      session_options = std::move(*provided_options);
    } else {
      // Configure default execution providers
      // Default to CPU execution provider
    }

    // Create a new session
    std::unique_ptr<Ort::Session> ort_session = std::make_unique<Ort::Session>(env_, model_path, session_options);

    // Create session info
    SessionInfo session_info;
    session_info.session = std::move(ort_session);

    // Get input names
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session_info.session->GetInputCount();
    session_info.input_names.clear();

    for (size_t i = 0; i < num_inputs; i++) {
      auto input_name = session_info.session->GetInputNameAllocated(i, allocator);
      session_info.input_names.push_back(std::string(input_name.get()));
    }

    // Get output names
    size_t num_outputs = session_info.session->GetOutputCount();
    session_info.output_names.clear();

    for (size_t i = 0; i < num_outputs; i++) {
      auto output_name = session_info.session->GetOutputNameAllocated(i, allocator);
      session_info.output_names.push_back(std::string(output_name.get()));
    }

    // Store the session info
    sessions_[session_id] = std::move(session_info);

    return session_id;
  } catch (const Ort::Exception &e) {
    std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
    // Return an empty string to indicate failure
    return "";
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return "";
  }
}

Ort::Session *SessionManager::getSession(const std::string &session_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = sessions_.find(session_id);
  if (it != sessions_.end()) {
    return it->second.session.get();
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