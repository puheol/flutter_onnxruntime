#ifndef FLUTTER_ONNXRUNTIME_SESSION_MANAGER_H_
#define FLUTTER_ONNXRUNTIME_SESSION_MANAGER_H_

#include "pch.h"

namespace flutter_onnxruntime {

// Manages ONNX Runtime sessions with proper resource handling
class SessionManager {
public:
  SessionManager();
  ~SessionManager();

  // Disallow copy and assign
  SessionManager(const SessionManager &) = delete;
  SessionManager &operator=(const SessionManager &) = delete;

  // Create a new session from model path
  std::string createSession(const std::string &modelPath, const std::map<std::string, std::string> &options);

  // Create a new session from model buffer
  std::string createSessionFromBuffer(const std::vector<uint8_t> &modelBuffer,
                                      const std::map<std::string, std::string> &options);

  // Get a session by ID
  Ort::Session *getSession(const std::string &sessionId);

  // Close and cleanup a session
  bool closeSession(const std::string &sessionId);

  // Get session metadata
  flutter::EncodableMap getSessionMetadata(const std::string &sessionId);

  // Get input info from a session
  flutter::EncodableMap getInputInfo(const std::string &sessionId);

  // Get output info from a session
  flutter::EncodableMap getOutputInfo(const std::string &sessionId);

  // Get a list of available execution providers
  flutter::EncodableList getAvailableExecutionProviders();

private:
  // Private implementation details
  std::unordered_map<std::string, std::unique_ptr<Ort::Session>> sessions_;
  std::mutex sessionsMutex_;
  Ort::Env env_;

  // Helper methods for configuring session options
  std::unique_ptr<Ort::SessionOptions> configureSessionOptions(const std::map<std::string, std::string> &options);

  // Generate a unique session ID
  std::string generateSessionId();
};

} // namespace flutter_onnxruntime

#endif // FLUTTER_ONNXRUNTIME_SESSION_MANAGER_H_