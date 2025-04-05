import Cocoa
import FlutterMacOS
import onnxruntime
import Foundation

public class FlutterOnnxruntimePlugin: NSObject, FlutterPlugin {
  private var sessions = [String: ORTSession]()
  private var env: ORTEnv?
  
  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "flutter_onnxruntime", binaryMessenger: registrar.messenger())
    let instance = FlutterOnnxruntimePlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    if env == nil {
      do {
        env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
      } catch {
        result(FlutterError(code: "ENV_INIT_FAILED", message: error.localizedDescription, details: nil))
        return
      }
    }
    
    switch call.method {
    case "getPlatformVersion":
      // Use macOS-specific system version info
      result("macOS " + ProcessInfo.processInfo.operatingSystemVersionString)
      
    case "createSession":
      guard let args = call.arguments as? [String: Any],
            let modelPath = args["modelPath"] as? String else {
        result(FlutterError(code: "INVALID_ARGS", message: "Model path is required", details: nil))
        return
      }
      
      do {
        let sessionOptions = try ORTSessionOptions()
        
        if let options = args["sessionOptions"] as? [String: Any] {
          if let intraOpNumThreads = options["intraOpNumThreads"] as? Int {
            try sessionOptions.setIntraOpNumThreads(Int32(intraOpNumThreads))
          }
          
          if let interOpNumThreads = options["interOpNumThreads"] as? Int {
            try sessionOptions.setInterOpNumThreads(Int32(interOpNumThreads))
          }
          
          if let enableCpuMemArena = options["enableCpuMemArena"] as? Bool {
            sessionOptions.enableCPUMemArena = enableCpuMemArena
          }
        }
        
        // Check if file exists
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: modelPath) {
          result(FlutterError(code: "FILE_NOT_FOUND", message: "Model file not found at path: \(modelPath)", details: nil))
          return
        }
        
        // Create session from file path
        let session = try ORTSession(env: env!, path: modelPath, sessionOptions: sessionOptions)
        let sessionId = UUID().uuidString
        sessions[sessionId] = session
        
        // Get input and output names
        var inputNames: [String] = []
        var outputNames: [String] = []
        
        for i in 0..<session.inputCount {
          if let name = try? session.inputName(at: i) {
            inputNames.append(name)
          }
        }
        
        for i in 0..<session.outputCount {
          if let name = try? session.outputName(at: i) {
            outputNames.append(name)
          }
        }
        
        result([
          "sessionId": sessionId,
          "inputNames": inputNames,
          "outputNames": outputNames
        ])
      } catch {
        result(FlutterError(code: "SESSION_CREATION_FAILED", message: error.localizedDescription, details: nil))
      }
      
    case "runInference":
      guard let args = call.arguments as? [String: Any],
            let sessionId = args["sessionId"] as? String,
            let inputs = args["inputs"] as? [String: Any] else {
        result(FlutterError(code: "INVALID_ARGS", message: "Session ID and inputs are required", details: nil))
        return
      }
      
      guard let session = sessions[sessionId] else {
        result(FlutterError(code: "INVALID_SESSION", message: "Session not found", details: nil))
        return
      }
      
      // Handle different input types and create ORTValue objects
      // This is simplified - you'll need more complex logic to handle
      // different tensor types and shapes
      
      // Run inference and convert results to a Flutter-compatible format
      // This is a placeholder for the actual implementation
      
      result(["outputs": [:]])
      
    case "closeSession":
      guard let args = call.arguments as? [String: Any],
            let sessionId = args["sessionId"] as? String else {
        result(FlutterError(code: "INVALID_ARGS", message: "Session ID is required", details: nil))
        return
      }
      
      if let _ = sessions.removeValue(forKey: sessionId) {
        result(nil)
      } else {
        result(FlutterError(code: "INVALID_SESSION", message: "Session not found", details: nil))
      }
      
    default:
      result(FlutterMethodNotImplemented)
    }
  }
}
