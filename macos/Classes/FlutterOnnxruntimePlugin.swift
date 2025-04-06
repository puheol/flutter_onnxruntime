import Cocoa
import FlutterMacOS
import onnxruntime_objc
import Foundation

enum OrtError: Error {
    case flutterError(FlutterError)
}

public class FlutterOnnxruntimePlugin: NSObject, FlutterPlugin {
  private var sessions = [String: ORTSession]()
  private var env: ORTEnv?
  
  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "flutter_onnxruntime", binaryMessenger: registrar.messenger)
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
    
    /** Create a new session

      Create a new session from a model file path.

      Reference: https://onnxruntime.ai/docs/api/objectivec/Classes/ORTSession.html
    */
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
          
          // if let interOpNumThreads = options["interOpNumThreads"] as? Int {
          //   try sessionOptions.setInterOpNumThreads(Int32(interOpNumThreads))
          // }
          
          // if let enableCpuMemArena = options["enableCpuMemArena"] as? Bool {
          //   sessionOptions.enableCPUMemArena = enableCpuMemArena
          // }
        }
        
        // Check if file exists
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: modelPath) {
          result(FlutterError(code: "FILE_NOT_FOUND", message: "Model file not found at path: \(modelPath)", details: nil))
          return
        }
        
        // Create session from file path
        let session = try ORTSession(env: env!, modelPath: modelPath, sessionOptions: sessionOptions)
        let sessionId = UUID().uuidString
        sessions[sessionId] = session
        
        // Get input and output names
        var inputNames: [String] = []
        var outputNames: [String] = []
        
        // Get input names
        if let inputNodeNames = try? session.inputNames() {
          inputNames = inputNodeNames
        }
        
        // Get output names
        if let outputNodeNames = try? session.outputNames() {
          outputNames = outputNodeNames
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
      print("runInference")
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

        // Get output names
        let outputNames = try session.outputNames()
        
        // Run inference
        let outputs = try session.run(withInputs: ortInputs, outputNames: Set(outputNames), runOptions: nil)

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
      
      guard sessions[sessionId] != nil else {
        result(FlutterError(code: "INVALID_SESSION", message: "Session not found", details: nil))
        return
      }
      
      // Return empty map as metadata functionality may not be available
      result([:])
      
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
        
        let inputNames = try session.inputNames()
        
        for name in inputNames {
          let infoMap: [String: Any] = ["name": name]
          nodeInfoList.append(infoMap)
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

        let outputNames = try session.outputNames()
        
        for name in outputNames {
          let infoMap: [String: Any] = ["name": name]
          nodeInfoList.append(infoMap)
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
        throw OrtError.flutterError(FlutterError(code: "INVALID_SHAPE", message: "Shape information required for input '\(name)'", details: nil))
      }
      
      // Create tensor based on input type
      if let floatArray = value as? [Float] {
        let data = NSMutableData(bytes: floatArray, length: floatArray.count * MemoryLayout<Float>.stride)
        let tensor = try ORTValue(tensorData: data, elementType: .float, shape: shape)
        ortInputs[name] = tensor
      } else if let intArray = value as? [Int32] {
        let data = NSMutableData(bytes: intArray, length: intArray.count * MemoryLayout<Int32>.stride)
        let tensor = try ORTValue(tensorData: data, elementType: .int32, shape: shape)
        ortInputs[name] = tensor
      } else if let doubleArray = value as? [Double] {
        // Convert double to float as ORTTensorElementDataType may not support double
        let floatArray = doubleArray.map { Float($0) }
        let data = NSMutableData(bytes: floatArray, length: floatArray.count * MemoryLayout<Float>.stride)
        let tensor = try ORTValue(tensorData: data, elementType: .float, shape: shape)
        ortInputs[name] = tensor
      } else if let numberArray = value as? [NSNumber] {
        // Convert NSNumber array to float array
        let floatArray = numberArray.map { $0.floatValue }
        let data = NSMutableData(bytes: floatArray, length: floatArray.count * MemoryLayout<Float>.stride)
        let tensor = try ORTValue(tensorData: data, elementType: .float, shape: shape)
        ortInputs[name] = tensor
      } else {
        throw OrtError.flutterError(FlutterError(code: "UNSUPPORTED_INPUT_TYPE", message: "Unsupported input type for '\(name)'", details: nil))
      }
    }
    
    return ortInputs
  }

  private func convertOutputsToFlutterFormat(outputs: [String: ORTValue], session: ORTSession) throws -> [String: Any] {
    var flutterOutputs: [String: Any] = [:]
    
    for (name, value) in outputs {
      if let tensorInfo = try? value.tensorTypeAndShapeInfo() {
        // Get shape information
        let shape = tensorInfo.shape.map { Int(truncating: $0) }
        flutterOutputs["\(name)_shape"] = shape
        
        // Calculate total element count
        let elementCount = shape.reduce(1, *)
 
        // Extract data based on tensor type
        switch tensorInfo.elementType {
        case .float:
          // For float tensors
          let dataPtr = try value.tensorData()
          let floatPtr = dataPtr.bytes.bindMemory(to: Float.self, capacity: elementCount)
          let floatBuffer = UnsafeBufferPointer(start: floatPtr, count: elementCount)
          flutterOutputs[name] = Array(floatBuffer)

        case .int8:
          // For int8 tensors
          let dataPtr = try value.tensorData()
          let int8Ptr = dataPtr.bytes.bindMemory(to: Int8.self, capacity: elementCount)
          let int8Buffer = UnsafeBufferPointer(start: int8Ptr, count: elementCount)
          flutterOutputs[name] = Array(int8Buffer)

        case .uInt8:
          // For uint8 tensors
          let dataPtr = try value.tensorData()
          let uint8Ptr = dataPtr.bytes.bindMemory(to: UInt8.self, capacity: elementCount)
          let uint8Buffer = UnsafeBufferPointer(start: uint8Ptr, count: elementCount)
          flutterOutputs[name] = Array(uint8Buffer)
          
        case .int32:
          // For int32 tensors
          let dataPtr = try value.tensorData()
          let intPtr = dataPtr.bytes.bindMemory(to: Int32.self, capacity: elementCount)
          let intBuffer = UnsafeBufferPointer(start: intPtr, count: elementCount)
          flutterOutputs[name] = Array(intBuffer)

        case .uInt32:
          // For uint32 tensors
          let dataPtr = try value.tensorData()
          let uint32Ptr = dataPtr.bytes.bindMemory(to: UInt32.self, capacity: elementCount)
          let uint32Buffer = UnsafeBufferPointer(start: uint32Ptr, count: elementCount)
          flutterOutputs[name] = Array(uint32Buffer)

        case .int64:
          // For int64 tensors
          let dataPtr = try value.tensorData()
          let int64Ptr = dataPtr.bytes.bindMemory(to: Int64.self, capacity: elementCount)
          let int64Buffer = UnsafeBufferPointer(start: int64Ptr, count: elementCount)
          flutterOutputs[name] = Array(int64Buffer)
        
        case .uInt64:
          // For uint64 tensors
          let dataPtr = try value.tensorData()
          let uint64Ptr = dataPtr.bytes.bindMemory(to: UInt64.self, capacity: elementCount)
          let uint64Buffer = UnsafeBufferPointer(start: uint64Ptr, count: elementCount)
          flutterOutputs[name] = Array(uint64Buffer)  
        
        case .string:
          // For string tensors
          let dataPtr = try value.tensorData()
          let stringPtr = dataPtr.bytes.bindMemory(to: String.self, capacity: elementCount)
          let stringBuffer = UnsafeBufferPointer(start: stringPtr, count: elementCount)
          flutterOutputs[name] = Array(stringBuffer)

        default:
          // Try extracting as float for other types
          do {
            let dataPtr = try value.tensorData()
            let floatPtr = dataPtr.bytes.bindMemory(to: Float.self, capacity: elementCount)
            let floatBuffer = UnsafeBufferPointer(start: floatPtr, count: elementCount)
            flutterOutputs[name] = Array(floatBuffer)
          } catch {
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
