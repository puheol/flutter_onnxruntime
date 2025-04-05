import Flutter
import UIKit
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
      result("iOS " + UIDevice.current.systemVersion)
      
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
      
      do {
        // Create an input map for the ORT session
        let ortInputs = try createORTValueInputs(inputs: inputs, session: session)
        
        // Run inference
        let outputs = try session.run(withInputs: ortInputs)
        
        // Process outputs to Flutter-compatible format
        let flutterOutputs = try convertOutputsToFlutterFormat(outputs: outputs, session: session)
        
        result(["outputs": flutterOutputs])
      } catch {
        result(FlutterError(code: "INFERENCE_ERROR", message: error.localizedDescription, details: nil))
      }
      
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
      
    case "getMetadata":
      guard let args = call.arguments as? [String: Any],
            let sessionId = args["sessionId"] as? String else {
        result(FlutterError(code: "INVALID_ARGS", message: "Session ID is required", details: nil))
        return
      }
      
      guard let session = sessions[sessionId] else {
        result(FlutterError(code: "INVALID_SESSION", message: "Session not found", details: nil))
        return
      }
      
      do {
        let modelMetadata = session.modelMetadata
        
        let metadataMap: [String: Any] = [
          "producerName": modelMetadata.producerName ?? "",
          "graphName": modelMetadata.graphName ?? "",
          "domain": modelMetadata.domain ?? "",
          "description": modelMetadata.description ?? "",
          "version": modelMetadata.version,
          "customMetadataMap": modelMetadata.customMetadata ?? [:]
        ]
        
        result(metadataMap)
      } catch {
        result(FlutterError(code: "METADATA_ERROR", message: error.localizedDescription, details: nil))
      }
      
    case "getInputInfo":
      guard let args = call.arguments as? [String: Any],
            let sessionId = args["sessionId"] as? String else {
        result(FlutterError(code: "INVALID_ARGS", message: "Session ID is required", details: nil))
        return
      }
      
      guard let session = sessions[sessionId] else {
        result(FlutterError(code: "INVALID_SESSION", message: "Session not found", details: nil))
        return
      }
      
      do {
        var nodeInfoList: [[String: Any]] = []
        
        for i in 0..<session.inputCount {
          if let name = try? session.inputName(at: i),
             let info = try? session.inputTypeInfo(at: i) {
            var infoMap: [String: Any] = ["name": name]
            
            // Handle tensor info if possible
            if let tensorInfo = info as? ORTTensorTypeAndShapeInfo {
              let shape = try tensorInfo.shape.map { Int($0) }
              infoMap["shape"] = shape
              infoMap["type"] = tensorInfo.elementType.description
            } else {
              infoMap["shape"] = []
            }
            
            nodeInfoList.append(infoMap)
          }
        }
        
        result(nodeInfoList)
      } catch {
        result(FlutterError(code: "INPUT_INFO_ERROR", message: error.localizedDescription, details: nil))
      }
      
    case "getOutputInfo":
      guard let args = call.arguments as? [String: Any],
            let sessionId = args["sessionId"] as? String else {
        result(FlutterError(code: "INVALID_ARGS", message: "Session ID is required", details: nil))
        return
      }
      
      guard let session = sessions[sessionId] else {
        result(FlutterError(code: "INVALID_SESSION", message: "Session not found", details: nil))
        return
      }
      
      do {
        var nodeInfoList: [[String: Any]] = []
        
        for i in 0..<session.outputCount {
          if let name = try? session.outputName(at: i),
             let info = try? session.outputTypeInfo(at: i) {
            var infoMap: [String: Any] = ["name": name]
            
            // Handle tensor info if possible
            if let tensorInfo = info as? ORTTensorTypeAndShapeInfo {
              let shape = try tensorInfo.shape.map { Int($0) }
              infoMap["shape"] = shape
              infoMap["type"] = tensorInfo.elementType.description
            } else {
              infoMap["shape"] = []
            }
            
            nodeInfoList.append(infoMap)
          }
        }
        
        result(nodeInfoList)
      } catch {
        result(FlutterError(code: "OUTPUT_INFO_ERROR", message: error.localizedDescription, details: nil))
      }
      
    default:
      result(FlutterMethodNotImplemented)
    }
  }

  // Helper functions

  private func createORTValueInputs(inputs: [String: Any], session: ORTSession) throws -> [String: ORTValue] {
    var ortInputs: [String: ORTValue] = [:]
    
    for (name, value) in inputs {
      // Skip shape info
      if name.hasSuffix("_shape") {
        continue
      }
      
      // Get shape if provided
      let shapeName = "\(name)_shape"
      let shape: [NSNumber]
      
      if let shapeArray = inputs[shapeName] as? [NSNumber] {
        shape = shapeArray
      } else if let value = value as? [Any] {
        // Default to 1D shape if not provided
        shape = [NSNumber(value: value.count)]
      } else {
        throw FlutterError(code: "INVALID_SHAPE", message: "Shape information required for input '\(name)'", details: nil)
      }
      
      // Create tensor based on input type
      if let floatArray = value as? [Float] {
        let tensor = try ORTValue.createTensor(fromArray: floatArray, shape: shape)
        ortInputs[name] = tensor
      } else if let intArray = value as? [Int32] {
        let tensor = try ORTValue.createTensor(fromArray: intArray, shape: shape)
        ortInputs[name] = tensor
      } else if let doubleArray = value as? [Double] {
        let tensor = try ORTValue.createTensor(fromArray: doubleArray, shape: shape)
        ortInputs[name] = tensor
      } else if let numberArray = value as? [NSNumber] {
        // Convert NSNumber array to float array
        let floatArray = numberArray.map { $0.floatValue }
        let tensor = try ORTValue.createTensor(fromArray: floatArray, shape: shape)
        ortInputs[name] = tensor
      } else {
        throw FlutterError(code: "UNSUPPORTED_INPUT_TYPE", message: "Unsupported input type for '\(name)'", details: nil)
      }
    }
    
    return ortInputs
  }

  private func convertOutputsToFlutterFormat(outputs: [String: ORTValue], session: ORTSession) throws -> [String: Any] {
    var flutterOutputs: [String: Any] = [:]
    
    for (name, value) in outputs {
      if let tensorInfo = try? value.tensorTypeAndShapeInfo {
        // Get shape information
        let shape = try tensorInfo.shape.map { Int($0) }
        flutterOutputs["\(name)_shape"] = shape
        
        // Extract data based on tensor type
        switch tensorInfo.elementType {
        case .float:
          if let data = try? value.floatArray() {
            flutterOutputs[name] = data
          }
        case .int:
          if let data = try? value.intArray() {
            flutterOutputs[name] = data
          }
        case .double:
          if let data = try? value.doubleArray() {
            // Convert to float for Flutter compatibility
            flutterOutputs[name] = data.map { Float($0) }
          }
        default:
          // For other types, try to extract as float (most common case)
          if let data = try? value.floatArray() {
            flutterOutputs[name] = data
          } else {
            flutterOutputs[name] = "Unsupported output tensor type: \(tensorInfo.elementType)"
          }
        }
      } else {
        flutterOutputs[name] = "Output is not a tensor"
      }
    }
    
    return flutterOutputs
  }
}
