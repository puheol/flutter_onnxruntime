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

  // swiftlint:disable:next cyclomatic_complexity
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
      handleCreateSession(call: call, result: result)
    case "runInference":
      handleRunInference(call: call, result: result)
    case "closeSession":
      handleCloseSession(call: call, result: result)
    case "getMetadata":
      handleGetMetadata(call: call, result: result)
    case "getInputInfo":
      handleGetInputInfo(call: call, result: result)
    case "getOutputInfo":
      handleGetOutputInfo(call: call, result: result)
    case "createOrtValue":
      handleCreateOrtValue(call, result: result)
    case "convertOrtValue":
      handleConvertOrtValue(call, result: result)
    case "moveOrtValueToDevice":
      handleMoveOrtValueToDevice(call, result: result)
    case "getOrtValueData":
      handleGetOrtValueData(call, result: result)
    case "releaseOrtValue":
      handleReleaseOrtValue(call, result: result)
    default:
      result(FlutterMethodNotImplemented)
    }
  }

  private func handleCreateSession(call: FlutterMethodCall, result: @escaping FlutterResult) {
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
      guard let safeEnv = env else {
        result(FlutterError(code: "ENV_NOT_INITIALIZED", message: "ONNX Runtime environment not initialized", details: nil))
        return
      }

      let session = try ORTSession(env: safeEnv, modelPath: modelPath, sessionOptions: sessionOptions)
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
  }

  private func handleRunInference(call: FlutterMethodCall, result: @escaping FlutterResult) {
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
  }

  private func handleCloseSession(call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let sessionId = args["sessionId"] as? String else {
      result(FlutterError(code: "INVALID_ARGS", message: "Session ID is required", details: nil))
      return
    }

    if sessions.removeValue(forKey: sessionId) != nil {
      result(nil)
    } else {
      result(FlutterError(code: "INVALID_SESSION", message: "Session not found", details: nil))
    }
  }

  private func handleGetMetadata(call: FlutterMethodCall, result: @escaping FlutterResult) {
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
  }

  private func handleGetInputInfo(call: FlutterMethodCall, result: @escaping FlutterResult) {
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
  }

  private func handleGetOutputInfo(call: FlutterMethodCall, result: @escaping FlutterResult) {
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
  }

  // Helper functions

  private func createORTValueInputs(inputs: [String: Any], session: ORTSession) throws -> [String: ORTValue] {
    var ortInputs: [String: ORTValue] = [:]

    for (name, value) in inputs {
      // Skip shape info
      if name.hasSuffix("_shape") {
        continue
      }

      // Check if the input is an OrtValue reference (sent as dictionary with valueId)
      if let valueDict = value as? [String: Any], let valueId = valueDict["valueId"] as? String {
        if let existingValue = ortValues[valueId] {
          ortInputs[name] = existingValue
          continue
        } else {
          throw OrtError.flutterError(FlutterError(code: "INVALID_ORT_VALUE", message: "OrtValue with ID \(valueId) not found", details: nil))
        }
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

  // swiftlint:disable:next cyclomatic_complexity
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

  // MARK: - OrtValue Management

  private var ortValues: [String: ONNXValue] = [:]
  
  private func handleCreateOrtValue(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let sourceType = args["sourceType"] as? String,
          let data = args["data"],
          let shape = args["shape"] as? [Int] else {
      result(FlutterError(code: "INVALID_ARGS", message: "Missing required arguments", details: nil))
      return
    }
    
    let targetType = args["targetType"] as? String ?? sourceType
    let device = args["device"] as? String ?? "cpu"
    
    // Convert shape to Int64 array for ONNX Runtime
    let shapeInt64 = shape.map { Int64($0) }
    
    do {
      // Create tensor based on source data type
      var tensor: ONNXValue
      
      switch sourceType {
      case "float32":
        if let floatArray = data as? [Float] {
          // Create float tensor
          tensor = try env?.createTensor(fromArray: floatArray, shape: shapeInt64)
        } else if let doubleArray = data as? [Double] {
          // Convert double to float
          let floatArray = doubleArray.map { Float($0) }
          tensor = try env?.createTensor(fromArray: floatArray, shape: shapeInt64)
        } else {
          result(FlutterError(code: "INVALID_DATA", message: "Data must be a list of numbers for float32 type", details: nil))
          return
        }
        
      case "int32":
        if let intArray = data as? [Int32] {
          tensor = try env?.createTensor(fromArray: intArray, shape: shapeInt64)
        } else if let intArray = data as? [Int] {
          let int32Array = intArray.map { Int32($0) }
          tensor = try env?.createTensor(fromArray: int32Array, shape: shapeInt64)
        } else {
          result(FlutterError(code: "INVALID_DATA", message: "Data must be a list of numbers for int32 type", details: nil))
          return
        }
        
      case "int64":
        if let longArray = data as? [Int64] {
          tensor = try env?.createTensor(fromArray: longArray, shape: shapeInt64)
        } else if let intArray = data as? [Int] {
          let int64Array = intArray.map { Int64($0) }
          tensor = try env?.createTensor(fromArray: int64Array, shape: shapeInt64)
        } else {
          result(FlutterError(code: "INVALID_DATA", message: "Data must be a list of numbers for int64 type", details: nil))
          return
        }
        
      case "uint8":
        if let uintArray = data as? [UInt8] {
          tensor = try env?.createTensor(fromArray: uintArray, shape: shapeInt64)
        } else if let intArray = data as? [Int] {
          let uintArray = intArray.map { UInt8($0) }
          tensor = try env?.createTensor(fromArray: uintArray, shape: shapeInt64)
        } else {
          result(FlutterError(code: "INVALID_DATA", message: "Data must be a list of numbers for uint8 type", details: nil))
          return
        }
        
      case "bool":
        if let boolArray = data as? [Bool] {
          tensor = try env?.createTensor(fromArray: boolArray, shape: shapeInt64)
        } else {
          result(FlutterError(code: "INVALID_DATA", message: "Data must be a list of booleans for bool type", details: nil))
          return
        }
        
      default:
        result(FlutterError(code: "UNSUPPORTED_TYPE", message: "Unsupported source data type: \(sourceType)", details: nil))
        return
      }
      
      // Perform type conversion if needed
      // Note: This is a placeholder. In a real implementation, you would implement
      // proper type conversion using ONNX Runtime APIs
      if targetType != sourceType {
        // This would be implemented based on available ONNX Runtime conversion APIs
        // For now, we'll just use the created tensor without conversion
      }
      
      // Generate unique ID for the tensor
      let valueId = UUID().uuidString
      ortValues[valueId] = tensor
      
      // Return tensor information
      let tensorInfo: [String: Any] = [
        "valueId": valueId,
        "dataType": targetType,
        "shape": shape,
        "device": device
      ]
      
      result(tensorInfo)
    } catch {
      result(FlutterError(code: "TENSOR_CREATION_ERROR", message: error.localizedDescription, details: nil))
    }
  }
  
  private func handleConvertOrtValue(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let valueId = args["valueId"] as? String,
          let targetType = args["targetType"] as? String else {
      result(FlutterError(code: "INVALID_ARGS", message: "Missing required arguments", details: nil))
      return
    }
    
    guard let tensor = ortValues[valueId] else {
      result(FlutterError(code: "INVALID_VALUE", message: "OrtValue with ID \(valueId) not found", details: nil))
      return
    }
    
    do {
      // Get tensor information
      let info = try tensor.info()
      let shape = info.shape
      
      // In a real implementation, this would perform the actual type conversion
      // For now, we'll just return the original tensor for most cases
      
      // Store the tensor with a new ID if it's different
      let newValueId = UUID().uuidString
      ortValues[newValueId] = tensor
      
      // Return tensor information
      let tensorInfo: [String: Any] = [
        "valueId": newValueId,
        "dataType": targetType,
        "shape": shape.map { Int($0) },
        "device": "cpu" // Assuming CPU device for now
      ]
      
      result(tensorInfo)
    } catch {
      result(FlutterError(code: "CONVERSION_ERROR", message: error.localizedDescription, details: nil))
    }
  }
  
  private func handleMoveOrtValueToDevice(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let valueId = args["valueId"] as? String,
          let targetDevice = args["targetDevice"] as? String else {
      result(FlutterError(code: "INVALID_ARGS", message: "Missing required arguments", details: nil))
      return
    }
    
    guard let tensor = ortValues[valueId] else {
      result(FlutterError(code: "INVALID_VALUE", message: "OrtValue with ID \(valueId) not found", details: nil))
      return
    }
    
    // Currently, we only support CPU device in this implementation
    if targetDevice != "cpu" {
      result(FlutterError(code: "UNSUPPORTED_DEVICE", message: "Only CPU device is supported in this implementation", details: nil))
      return
    }
    
    do {
      // Get tensor information
      let info = try tensor.info()
      let shape = info.shape
      
      // Return tensor information (no actual device transfer)
      let tensorInfo: [String: Any] = [
        "valueId": valueId,
        "dataType": info.elementDataType.description.lowercased(),
        "shape": shape.map { Int($0) },
        "device": "cpu"
      ]
      
      result(tensorInfo)
    } catch {
      result(FlutterError(code: "DEVICE_TRANSFER_ERROR", message: error.localizedDescription, details: nil))
    }
  }
  
  private func handleGetOrtValueData(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let valueId = args["valueId"] as? String,
          let dataType = args["dataType"] as? String else {
      result(FlutterError(code: "INVALID_ARGS", message: "Missing required arguments", details: nil))
      return
    }
    
    guard let tensor = ortValues[valueId] else {
      result(FlutterError(code: "INVALID_VALUE", message: "OrtValue with ID \(valueId) not found", details: nil))
      return
    }
    
    do {
      // Get tensor information
      let info = try tensor.info()
      let shape = info.shape
      
      // Extract data according to requested type
      var data: Any
      
      // Extract data based on the tensor's type and requested type
      // This is a simplified version - a complete implementation would handle
      // all possible source types and conversions
      switch dataType {
      case "float32":
        if info.elementDataType == .float {
          // Get float data directly
          let floatData = try tensor.floatData()
          data = Array(floatData)
        } else if info.elementDataType == .int32 {
          // Convert int32 to float32
          let intData = try tensor.int32Data()
          data = intData.map { Float($0) }
        } else if info.elementDataType == .int64 {
          // Convert int64 to float32
          let longData = try tensor.int64Data()
          data = longData.map { Float($0) }
        } else {
          result(FlutterError(code: "CONVERSION_ERROR", message: "Cannot convert to float32", details: nil))
          return
        }
        
      case "int32":
        if info.elementDataType == .int32 {
          // Get int32 data directly
          let intData = try tensor.int32Data()
          data = Array(intData)
        } else if info.elementDataType == .float {
          // Convert float to int32
          let floatData = try tensor.floatData()
          data = floatData.map { Int32($0) }
        } else if info.elementDataType == .int64 {
          // Convert int64 to int32
          let longData = try tensor.int64Data()
          data = longData.map { Int32($0) }
        } else {
          result(FlutterError(code: "CONVERSION_ERROR", message: "Cannot convert to int32", details: nil))
          return
        }
        
      case "int64":
        if info.elementDataType == .int64 {
          // Get int64 data directly
          let longData = try tensor.int64Data()
          data = Array(longData)
        } else if info.elementDataType == .float {
          // Convert float to int64
          let floatData = try tensor.floatData()
          data = floatData.map { Int64($0) }
        } else if info.elementDataType == .int32 {
          // Convert int32 to int64
          let intData = try tensor.int32Data()
          data = intData.map { Int64($0) }
        } else {
          result(FlutterError(code: "CONVERSION_ERROR", message: "Cannot convert to int64", details: nil))
          return
        }
        
      case "uint8":
        if info.elementDataType == .uint8 {
          // Get uint8 data directly
          let byteData = try tensor.uint8Data()
          data = Array(byteData)
        } else {
          result(FlutterError(code: "CONVERSION_ERROR", message: "Cannot convert to uint8", details: nil))
          return
        }
        
      case "bool":
        if info.elementDataType == .bool {
          // Get bool data directly
          let boolData = try tensor.boolData()
          data = Array(boolData)
        } else {
          result(FlutterError(code: "CONVERSION_ERROR", message: "Cannot convert to bool", details: nil))
          return
        }
        
      default:
        result(FlutterError(code: "UNSUPPORTED_TYPE", message: "Unsupported data type: \(dataType)", details: nil))
        return
      }
      
      // Return data with shape
      let resultMap: [String: Any] = [
        "data": data,
        "shape": shape.map { Int($0) }
      ]
      
      result(resultMap)
    } catch {
      result(FlutterError(code: "DATA_EXTRACTION_ERROR", message: error.localizedDescription, details: nil))
    }
  }
  
  private func handleReleaseOrtValue(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let valueId = args["valueId"] as? String else {
      result(FlutterError(code: "INVALID_ARGS", message: "Missing value ID", details: nil))
      return
    }
    
    // Remove and release tensor
    ortValues.removeValue(forKey: valueId)
    
    result(nil)
  }
}
