// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "session_manager.h"
#include "windows_utils.h"

namespace flutter_onnxruntime {

// Initialize the SessionManager
SessionManager::SessionManager() {
  // Initialize ONNX Runtime environment
  env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "flutter_onnxruntime");
}

// Clean up resources
SessionManager::~SessionManager() {
  // Clean up sessions with a lock to ensure thread safety
  std::lock_guard<std::mutex> lock(sessionsMutex_);
  sessions_.clear();
}

// This is a placeholder for the createSession function
// It will be properly implemented in the next iteration
std::string SessionManager::createSession(const std::string &modelPath, const flutter::EncodableMap &sessionOptions) {

  // For now, just log the action and return a dummy session ID
  // This will be replaced with a proper implementation

  // Normalize path for Windows
  std::string normalizedPath = WindowsUtils::normalizePathSeparators(modelPath);

  // Check if model exists
  if (!WindowsUtils::pathExists(normalizedPath)) {
    throw std::runtime_error("Model file not found: " + normalizedPath);
  }

  // Generate a dummy session ID
  std::string sessionId = "session_placeholder";

  return sessionId;
}

// Placeholder for run inference
flutter::EncodableMap SessionManager::runInference(const std::string &sessionId,
                                                   const std::map<std::string, Ort::Value *> &inputs,
                                                   const flutter::EncodableMap &runOptions) {

  // For now, just return an empty map
  // This will be replaced with a proper implementation
  flutter::EncodableMap result;
  result[flutter::EncodableValue("status")] = flutter::EncodableValue("placeholder");

  return result;
}

// Placeholder for close session
bool SessionManager::closeSession(const std::string &sessionId) {
  // For now, just return true
  // This will be replaced with a proper implementation
  return true;
}

// Placeholder for getting metadata
flutter::EncodableMap SessionManager::getMetadata(const std::string &sessionId) {
  // For now, just return an empty map
  // This will be replaced with a proper implementation
  flutter::EncodableMap metadata;
  metadata[flutter::EncodableValue("note")] = flutter::EncodableValue("Placeholder implementation");

  return metadata;
}

// Placeholder for getting input info
flutter::EncodableList SessionManager::getInputInfo(const std::string &sessionId) {
  // For now, just return an empty list
  // This will be replaced with a proper implementation
  return flutter::EncodableList();
}

// Placeholder for getting output info
flutter::EncodableList SessionManager::getOutputInfo(const std::string &sessionId) {
  // For now, just return an empty list
  // This will be replaced with a proper implementation
  return flutter::EncodableList();
}

} // namespace flutter_onnxruntime