package com.masicai.flutteronnxruntime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.util.Log
import androidx.annotation.NonNull
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import java.io.File
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

/** FlutterOnnxruntimePlugin */
class FlutterOnnxruntimePlugin : FlutterPlugin, MethodCallHandler {
    /** The MethodChannel that will the communication between Flutter and native Android

     This local reference serves to register the plugin with the Flutter Engine and unregister it
     when the Flutter Engine is detached from the Activity
     */
    private lateinit var channel: MethodChannel
    private lateinit var ortEnvironment: OrtEnvironment
    private val sessions = ConcurrentHashMap<String, OrtSession>()

    // Store OrtValues (tensors) by ID
    private val ortValues = ConcurrentHashMap<String, OnnxValue>()

    override fun onAttachedToEngine(
        @NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding,
    ) {
        channel = MethodChannel(flutterPluginBinding.binaryMessenger, "flutter_onnxruntime")
        channel.setMethodCallHandler(this)
        ortEnvironment = OrtEnvironment.getEnvironment()
    }

    override fun onMethodCall(
        @NonNull call: MethodCall,
        @NonNull result: Result,
    ) {
        when (call.method) {
            "getPlatformVersion" -> {
                result.success("Android ${android.os.Build.VERSION.RELEASE}")
            }
            "createSession" -> {
                try {
                    val modelPath = call.argument<String>("modelPath")
                    val sessionOptions = call.argument<Map<String, Any>>("sessionOptions") ?: emptyMap()

                    if (modelPath == null) {
                        result.error("NULL_MODEL_PATH", "Model path cannot be null", null)
                        return
                    }

                    val ortSessionOptions = OrtSession.SessionOptions()

                    // Configure session options based on the provided map
                    if (sessionOptions.containsKey("intraOpNumThreads")) {
                        ortSessionOptions.setIntraOpNumThreads((sessionOptions["intraOpNumThreads"] as Number).toInt())
                    }

                    if (sessionOptions.containsKey("interOpNumThreads")) {
                        ortSessionOptions.setInterOpNumThreads((sessionOptions["interOpNumThreads"] as Number).toInt())
                    }

                    // Load model from file path
                    val modelFile = File(modelPath)
                    if (!modelFile.exists()) {
                        result.error("FILE_NOT_FOUND", "Model file not found at path: $modelPath", null)
                        return
                    }

                    val session = ortEnvironment.createSession(modelPath, ortSessionOptions)
                    val sessionId = UUID.randomUUID().toString()
                    sessions[sessionId] = session

                    // Get input and output names
                    val inputNames = session.inputNames.toList()
                    val outputNames = session.outputNames.toList()

                    result.success(
                        mapOf(
                            "sessionId" to sessionId,
                            "inputNames" to inputNames,
                            "outputNames" to outputNames,
                        ),
                    )
                } catch (e: OrtException) {
                    result.error("ORT_ERROR", e.message, e.stackTraceToString())
                } catch (e: Exception) {
                    result.error("GENERIC_ERROR", e.message, e.stackTraceToString())
                }
            }
            "runInference" -> {
                try {
                    val sessionId = call.argument<String>("sessionId")
                    val inputs = call.argument<Map<String, Any>>("inputs")

                    if (sessionId == null || !sessions.containsKey(sessionId)) {
                        result.error("INVALID_SESSION", "Session not found", null)
                        return
                    }

                    if (inputs == null) {
                        result.error("NULL_INPUTS", "Inputs cannot be null", null)
                        return
                    }

                    val session = sessions[sessionId]!!
                    val ortInputs = HashMap<String, OnnxValue>()

                    try {
                        // Process inputs that aren't shape information
                        for ((name, value) in inputs) {
                            if (name.endsWith("_shape")) continue

                            // Get shape information if provided
                            val shapeName = "${name}_shape"
                            val shape =
                                if (inputs.containsKey(shapeName) && inputs[shapeName] is List<*>) {
                                    (inputs[shapeName] as List<*>).map { (it as Number).toLong() }.toLongArray()
                                } else {
                                    // Default shape for 1D data
                                    when (value) {
                                        is List<*> -> longArrayOf(value.size.toLong())
                                        is ByteArray -> longArrayOf(value.size.toLong())
                                        is FloatArray -> longArrayOf(value.size.toLong())
                                        is IntArray -> longArrayOf(value.size.toLong())
                                        else -> {
                                            result.error(
                                                "MISSING_SHAPE",
                                                "Shape information required for input '$name' of type: ${value?.javaClass?.name}",
                                                null,
                                            )
                                            return
                                        }
                                    }
                                }

                            // Create appropriate tensor based on input type
                            when (value) {
                                // Handle OrtValue reference (sent as Map with valueId)
                                is Map<*, *> -> {
                                    if (value.containsKey("valueId")) {
                                        val valueId = value["valueId"] as String
                                        val existingTensor = ortValues[valueId]
                                        if (existingTensor != null) {
                                            ortInputs[name] = existingTensor
                                        } else {
                                            result.error(
                                                "INVALID_ORT_VALUE",
                                                "OrtValue with ID $valueId not found",
                                                null,
                                            )
                                            return
                                        }
                                    } else {
                                        result.error(
                                            "INVALID_INPUT_FORMAT",
                                            "Input map for '$name' does not contain a valueId",
                                            null,
                                        )
                                        return
                                    }
                                }
                                is List<*> -> {
                                    if (value.isEmpty()) {
                                        result.error("EMPTY_INPUT", "Input '$name' is empty", null)
                                        return
                                    }

                                    // Handle based on element type
                                    when (val firstElement = value[0]) {
                                        is Number -> {
                                            val floatArray = value.map { (it as Number).toFloat() }.toFloatArray()
                                            val tensor = OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(floatArray), shape)
                                            ortInputs[name] = tensor
                                        }
                                        else -> {
                                            result.error(
                                                "UNSUPPORTED_INPUT_TYPE",
                                                "Unsupported input element type for '$name': ${firstElement?.javaClass?.name}",
                                                null,
                                            )
                                            return
                                        }
                                    }
                                }
                                is ByteArray -> {
                                    val tensor = OnnxTensor.createTensor(ortEnvironment, ByteBuffer.wrap(value), shape)
                                    ortInputs[name] = tensor
                                }
                                is FloatArray -> {
                                    val tensor = OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(value), shape)
                                    ortInputs[name] = tensor
                                }
                                is IntArray -> {
                                    val tensor = OnnxTensor.createTensor(ortEnvironment, IntBuffer.wrap(value), shape)
                                    ortInputs[name] = tensor
                                }
                                else -> {
                                    result.error(
                                        "UNSUPPORTED_INPUT_TYPE",
                                        "Unsupported input type for '$name': ${value?.javaClass?.name}",
                                        null,
                                    )
                                    return
                                }
                            }
                        }

                        // Convert inputs to the required type for session.run
                        val runInputs = HashMap<String, OnnxTensor>()
                        for ((name, value) in ortInputs) {
                            if (value is OnnxTensor) {
                                runInputs[name] = value
                            }
                        }

                        // Run inference with correctly typed inputs
                        val ortOutputs = session.run(runInputs)

                        // Process outputs
                        val outputs = HashMap<String, Any>()

                        // Convert tensor outputs to Flutter-compatible types
                        for (outputName in session.outputNames) {
                            val outputValue = ortOutputs[outputName]

                            Log.d("outputValue", outputValue.toString())

                            // Output tensor is wrapped in Optional[] for safety, unwrap the Optional if needed
                            val actualTensor =
                                when {
                                    outputValue.toString().startsWith("Optional[") -> {
                                        try {
                                            // Try to use the get() method if available
                                            val getMethod = outputValue.javaClass.getMethod("get")
                                            getMethod.invoke(outputValue) as? OnnxTensor
                                        } catch (e: Exception) {
                                            try {
                                                // Fallback to orElse(null) method
                                                val orElseMethod = outputValue.javaClass.getMethod("orElse", Object::class.java)
                                                orElseMethod.invoke(outputValue, null) as? OnnxTensor
                                            } catch (e2: Exception) {
                                                Log.e("ORT_ERROR", "Failed to unwrap Optional: ${e2.message}")
                                                null
                                            }
                                        }
                                    }
                                    outputValue is OnnxTensor -> outputValue
                                    else -> null
                                }

                            if (actualTensor != null) {
                                // Get shape information
                                val shape = actualTensor.info.shape
                                outputs["${outputName}_shape"] = shape.toList()

                                // Try to determine the tensor type and extract the appropriate data
                                try {
                                    // Try float - most common for ML model outputs
                                    val flatSize = shape.fold(1L) { acc, dim -> acc * dim }.toInt()
                                    val floatArray = FloatArray(flatSize)
                                    actualTensor.floatBuffer.get(floatArray)
                                    outputs[outputName] = floatArray.toList()
                                } catch (e: Exception) {
                                    try {
                                        // Try int
                                        val flatSize = shape.fold(1L) { acc, dim -> acc * dim }.toInt()
                                        val intArray = IntArray(flatSize)
                                        actualTensor.intBuffer.get(intArray)
                                        outputs[outputName] = intArray.toList()
                                    } catch (e2: Exception) {
                                        try {
                                            // Try long
                                            val flatSize = shape.fold(1L) { acc, dim -> acc * dim }.toInt()
                                            val longArray = LongArray(flatSize)
                                            actualTensor.longBuffer.get(longArray)
                                            outputs[outputName] = longArray.toList()
                                        } catch (e3: Exception) {
                                            try {
                                                // Try byte
                                                val flatSize = shape.fold(1L) { acc, dim -> acc * dim }.toInt()
                                                val byteArray = ByteArray(flatSize)
                                                actualTensor.byteBuffer.get(byteArray)
                                                outputs[outputName] = byteArray.map { it.toInt() and 0xFF }
                                            } catch (e4: Exception) {
                                                outputs[outputName] = "Failed to extract tensor data: ${e4.message}"
                                            }
                                        }
                                    }
                                }
                            } else {
                                outputs[outputName] = "Output is null or not a tensor: ${outputValue?.javaClass?.name}"
                            }
                        }

                        // Clean up
                        for (input in ortInputs.values) {
                            input.close()
                        }
                        // ortOutputs.iterator().forEach { key, value -> value.close() }

                        result.success(mapOf("outputs" to outputs))
                    } catch (e: Exception) {
                        // Clean up in case of error
                        for (input in ortInputs.values) {
                            input.close()
                        }
                        throw e
                    }
                } catch (e: OrtException) {
                    result.error("ORT_ERROR", e.message, e.stackTraceToString())
                } catch (e: Exception) {
                    result.error("GENERIC_ERROR", e.message, e.stackTraceToString())
                }
            }
            "closeSession" -> {
                try {
                    val sessionId = call.argument<String>("sessionId")

                    if (sessionId == null || !sessions.containsKey(sessionId)) {
                        result.error("INVALID_SESSION", "Session not found", null)
                        return
                    }

                    val session = sessions[sessionId]!!
                    session.close()
                    sessions.remove(sessionId)

                    result.success(null)
                } catch (e: Exception) {
                    result.error("GENERIC_ERROR", e.message, e.stackTraceToString())
                }
            }
            /** Get metadata about the model

             Returns metadata about the model such as producer name, graph name, domain, description, version, and custom metadata.

             Reference: https://onnxruntime.ai/docs/api/java/ai/onnxruntime/OrtSession.html#getMetadata()
             */
            "getMetadata" -> {
                try {
                    val sessionId = call.argument<String>("sessionId")

                    if (sessionId == null || !sessions.containsKey(sessionId)) {
                        result.error("INVALID_SESSION", "Session not found", null)
                        return
                    }

                    val session = sessions[sessionId]!!
                    val metadata = session.getMetadata()

                    // Convert custom metadata map to a standard Map
                    val customMetadataMap = metadata.customMetadata

                    val metadataMap =
                        mapOf(
                            "producerName" to metadata.producerName,
                            "graphName" to metadata.graphName,
                            "domain" to metadata.domain,
                            "description" to metadata.description,
                            "version" to metadata.version,
                            "customMetadataMap" to customMetadataMap,
                        )

                    result.success(metadataMap)
                } catch (e: Exception) {
                    result.error("GENERIC_ERROR", e.message, e.stackTraceToString())
                }
            }
            /** Get input info about the model

             Returns information about the model's inputs such as name, type, and shape.

             Reference: https://onnxruntime.ai/docs/api/java/ai/onnxruntime/OrtSession.html#getInputInfo()
             */
            "getInputInfo" -> {
                try {
                    val sessionId = call.argument<String>("sessionId")

                    if (sessionId == null || !sessions.containsKey(sessionId)) {
                        result.error("INVALID_SESSION", "Session not found", null)
                        return
                    }

                    val session = sessions[sessionId]!!
                    val nodeInfoList = ArrayList<Map<String, Any>>()

                    // Get all input info as Map<String, NodeInfo>
                    val inputInfoMap = session.getInputInfo()

                    // Convert to a list of maps for Flutter
                    for ((name, nodeInfo) in inputInfoMap) {
                        val infoMap = HashMap<String, Any>()
                        infoMap["name"] = name

                        // Get the info object and check its type
                        val info = nodeInfo.info

                        // Only extract shape if it's a TensorInfo
                        if (info is ai.onnxruntime.TensorInfo) {
                            val shape = info.shape
                            infoMap["shape"] = shape.toList()
                            infoMap["type"] = info.type.toString()
                        } else {
                            // For non-tensor types, provide an empty shape
                            infoMap["shape"] = emptyList<Long>()
                        }

                        nodeInfoList.add(infoMap)
                    }

                    result.success(nodeInfoList)
                } catch (e: Exception) {
                    result.error("GENERIC_ERROR", e.message, e.stackTraceToString())
                }
            }
            /** Get output info about the model

             Returns information about the model's outputs such as name, type, and shape.

             Reference: https://onnxruntime.ai/docs/api/java/ai/onnxruntime/OrtSession.html#getOutputInfo()
             */
            "getOutputInfo" -> {
                try {
                    val sessionId = call.argument<String>("sessionId")

                    if (sessionId == null || !sessions.containsKey(sessionId)) {
                        result.error("INVALID_SESSION", "Session not found", null)
                        return
                    }

                    val session = sessions[sessionId]!!
                    val nodeInfoList = ArrayList<Map<String, Any>>()

                    // Get all output info as Map<String, NodeInfo>
                    val outputInfoMap = session.getOutputInfo()

                    // Convert to a list of maps for Flutter
                    for ((name, nodeInfo) in outputInfoMap) {
                        val infoMap = HashMap<String, Any>()
                        infoMap["name"] = name

                        // Get the info object and check its type
                        val info = nodeInfo.info

                        // Only extract shape if it's a TensorInfo
                        if (info is ai.onnxruntime.TensorInfo) {
                            val shape = info.shape
                            infoMap["shape"] = shape.toList()
                            infoMap["type"] = info.type.toString()
                        } else {
                            // For non-tensor types, provide an empty shape
                            infoMap["shape"] = emptyList<Long>()
                        }

                        nodeInfoList.add(infoMap)
                    }

                    result.success(nodeInfoList)
                } catch (e: Exception) {
                    result.error("GENERIC_ERROR", e.message, e.stackTraceToString())
                }
            }
            // OrtValue methods
            "createOrtValue" -> {
                try {
                    val sourceType = call.argument<String>("sourceType")
                    val data = call.argument<Any>("data")
                    val shape = call.argument<List<Int>>("shape")
                    val targetType = call.argument<String>("targetType")
                    val device = call.argument<String>("device")

                    if (sourceType == null || data == null || shape == null) {
                        result.error("INVALID_ARGS", "Missing required arguments", null)
                        return
                    }

                    // Convert shape to long array for OnnxRuntime
                    val longShape = shape.map { it.toLong() }.toLongArray()

                    // Create tensor based on source data type
                    val tensor =
                        when (sourceType) {
                            "float32" -> {
                                val floatData =
                                    when (data) {
                                        is List<*> -> data.map { (it as Number).toFloat() }.toFloatArray()
                                        is FloatArray -> data
                                        else -> {
                                            result.error("INVALID_DATA", "Data must be a list of numbers for float32 type", null)
                                            return
                                        }
                                    }
                                OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(floatData), longShape)
                            }
                            "int32" -> {
                                val intData =
                                    when (data) {
                                        is List<*> -> data.map { (it as Number).toInt() }.toIntArray()
                                        is IntArray -> data
                                        else -> {
                                            result.error("INVALID_DATA", "Data must be a list of numbers for int32 type", null)
                                            return
                                        }
                                    }
                                OnnxTensor.createTensor(ortEnvironment, IntBuffer.wrap(intData), longShape)
                            }
                            "int64" -> {
                                val longData =
                                    when (data) {
                                        is List<*> -> data.map { (it as Number).toLong() }.toLongArray()
                                        is LongArray -> data
                                        else -> {
                                            result.error("INVALID_DATA", "Data must be a list of numbers for int64 type", null)
                                            return
                                        }
                                    }
                                OnnxTensor.createTensor(ortEnvironment, LongBuffer.wrap(longData), longShape)
                            }
                            "uint8" -> {
                                val byteData =
                                    when (data) {
                                        is List<*> -> {
                                            val bytes = ByteArray(data.size)
                                            for (i in data.indices) {
                                                bytes[i] = (data[i] as Number).toByte()
                                            }
                                            bytes
                                        }
                                        is ByteArray -> data
                                        else -> {
                                            result.error("INVALID_DATA", "Data must be a list of numbers for uint8 type", null)
                                            return
                                        }
                                    }
                                OnnxTensor.createTensor(ortEnvironment, ByteBuffer.wrap(byteData), longShape)
                            }
                            "bool" -> {
                                val boolData =
                                    when (data) {
                                        is List<*> -> {
                                            val bytes = ByteArray(data.size)
                                            for (i in data.indices) {
                                                bytes[i] = if (data[i] as Boolean) 1.toByte() else 0.toByte()
                                            }
                                            bytes
                                        }
                                        else -> {
                                            result.error("INVALID_DATA", "Data must be a list of booleans for bool type", null)
                                            return
                                        }
                                    }
                                // Boolean tensors are stored as bytes in ONNX Runtime
                                OnnxTensor.createTensor(ortEnvironment, ByteBuffer.wrap(boolData), longShape)
                            }
                            else -> {
                                result.error("UNSUPPORTED_TYPE", "Unsupported source data type: $sourceType", null)
                                return
                            }
                        }

                    // If target type is different from source type, perform conversion
                    // Note: This is a simplified example. In a real implementation, you would need
                    // to implement type conversion logic based on ONNX Runtime capabilities.
                    // For now, we'll just use the created tensor without conversion.

                    // Store the tensor with a unique ID
                    val valueId = UUID.randomUUID().toString()
                    ortValues[valueId] = tensor

                    // Return tensor information
                    val tensorInfo =
                        mapOf(
                            "valueId" to valueId,
                            "dataType" to (targetType ?: sourceType),
                            "shape" to shape,
                            "device" to (device ?: "cpu"),
                        )

                    result.success(tensorInfo)
                } catch (e: Exception) {
                    result.error("TENSOR_CREATION_ERROR", e.message, e.stackTraceToString())
                }
            }
            "convertOrtValue" -> {
                try {
                    val valueId = call.argument<String>("valueId")
                    val targetType = call.argument<String>("targetType")

                    if (valueId == null || targetType == null) {
                        result.error("INVALID_ARGS", "Missing required arguments", null)
                        return
                    }

                    val tensor = ortValues[valueId]
                    if (tensor == null) {
                        result.error("INVALID_VALUE", "OrtValue with ID $valueId not found", null)
                        return
                    }

                    if (tensor !is OnnxTensor) {
                        result.error("INVALID_TENSOR_TYPE", "OrtValue is not a tensor", null)
                        return
                    }

                    // Get tensor information
                    val shape = tensor.info.shape
                    val dataType = tensor.info.type.toString()

                    // For now, we'll implement a simple conversion for certain type pairs
                    // A full implementation would handle all possible conversions
                    val newTensor =
                        when {
                            // Float32 to other types
                            dataType == "FLOAT" && targetType == "float16" -> {
                                // Real implementation would convert float32 to float16
                                // This is a placeholder - just using the same tensor
                                tensor
                            }
                            // Int32 to Float32
                            dataType == "INT32" && targetType == "float32" -> {
                                val intBuffer = tensor.intBuffer
                                val intArray = IntArray(intBuffer.remaining())
                                intBuffer.get(intArray)

                                val floatArray = FloatArray(intArray.size) { intArray[it].toFloat() }
                                OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(floatArray), shape)
                            }
                            // Other conversions would be implemented similarly
                            else -> {
                                // If no conversion is implemented, just return the original tensor
                                tensor
                            }
                        }

                    // Store the new tensor if it's different from the original
                    val newValueId =
                        if (newTensor !== tensor) {
                            val id = UUID.randomUUID().toString()
                            ortValues[id] = newTensor
                            id
                        } else {
                            valueId
                        }

                    // Return tensor information
                    // Assuming CPU device for now
                    val tensorInfo =
                        mapOf(
                            "valueId" to newValueId,
                            "dataType" to targetType,
                            "shape" to shape.toList(),
                            "device" to "cpu",
                        )

                    result.success(tensorInfo)
                } catch (e: Exception) {
                    result.error("CONVERSION_ERROR", e.message, e.stackTraceToString())
                }
            }
            "moveOrtValueToDevice" -> {
                try {
                    val valueId = call.argument<String>("valueId")
                    val targetDevice = call.argument<String>("targetDevice")

                    if (valueId == null || targetDevice == null) {
                        result.error("INVALID_ARGS", "Missing required arguments", null)
                        return
                    }

                    val tensor = ortValues[valueId]
                    if (tensor == null) {
                        result.error("INVALID_VALUE", "OrtValue with ID $valueId not found", null)
                        return
                    }

                    // Currently, we only support CPU device in this implementation
                    // For GPU support, you would need to implement device transfer logic
                    if (targetDevice != "cpu") {
                        result.error("UNSUPPORTED_DEVICE", "Only CPU device is supported in this implementation", null)
                        return
                    }

                    // Return the same tensor since we're already on CPU
                    val tensorInfo =
                        mapOf(
                            "valueId" to valueId,
                            "dataType" to (if (tensor is OnnxTensor) tensor.info.type.toString().lowercase() else "unknown"),
                            "shape" to (if (tensor is OnnxTensor) tensor.info.shape.toList() else emptyList<Int>()),
                            "device" to "cpu",
                        )

                    result.success(tensorInfo)
                } catch (e: Exception) {
                    result.error("DEVICE_TRANSFER_ERROR", e.message, e.stackTraceToString())
                }
            }
            "getOrtValueData" -> {
                try {
                    val valueId = call.argument<String>("valueId")
                    val dataType = call.argument<String>("dataType")

                    if (valueId == null) {
                        result.error("INVALID_ARGS", "Missing value ID", null)
                        return
                    }

                    val tensor = ortValues[valueId]
                    if (tensor == null) {
                        result.error("INVALID_VALUE", "OrtValue with ID $valueId not found", null)
                        return
                    }

                    if (tensor !is OnnxTensor) {
                        result.error("INVALID_TENSOR_TYPE", "OrtValue is not a tensor", null)
                        return
                    }

                    // Get tensor shape
                    val shape = tensor.info.shape
                    val flatSize = shape.fold(1L) { acc, dim -> acc * dim }.toInt()

                    // Extract data according to requested type
                    val data =
                        when (dataType) {
                            "float32" -> {
                                try {
                                    val floatArray = FloatArray(flatSize)
                                    tensor.floatBuffer.get(floatArray)
                                    floatArray.toList()
                                } catch (e: Exception) {
                                    // If can't get as float directly, try conversion
                                    when (tensor.info.type.toString()) {
                                        "INT32" -> {
                                            val intArray = IntArray(flatSize)
                                            tensor.intBuffer.get(intArray)
                                            intArray.map { it.toFloat() }
                                        }
                                        "INT64" -> {
                                            val longArray = LongArray(flatSize)
                                            tensor.longBuffer.get(longArray)
                                            longArray.map { it.toFloat() }
                                        }
                                        else -> {
                                            result.error("CONVERSION_ERROR", "Cannot convert to float32", null)
                                            return
                                        }
                                    }
                                }
                            }
                            "int32" -> {
                                try {
                                    val intArray = IntArray(flatSize)
                                    tensor.intBuffer.get(intArray)
                                    intArray.toList()
                                } catch (e: Exception) {
                                    // Try conversion if direct extraction fails
                                    when (tensor.info.type.toString()) {
                                        "FLOAT" -> {
                                            val floatArray = FloatArray(flatSize)
                                            tensor.floatBuffer.get(floatArray)
                                            floatArray.map { it.toInt() }
                                        }
                                        "INT64" -> {
                                            val longArray = LongArray(flatSize)
                                            tensor.longBuffer.get(longArray)
                                            longArray.map { it.toInt() }
                                        }
                                        else -> {
                                            result.error("CONVERSION_ERROR", "Cannot convert to int32", null)
                                            return
                                        }
                                    }
                                }
                            }
                            "int64" -> {
                                try {
                                    val longArray = LongArray(flatSize)
                                    tensor.longBuffer.get(longArray)
                                    longArray.toList()
                                } catch (e: Exception) {
                                    // Try conversion
                                    when (tensor.info.type.toString()) {
                                        "FLOAT" -> {
                                            val floatArray = FloatArray(flatSize)
                                            tensor.floatBuffer.get(floatArray)
                                            floatArray.map { it.toLong() }
                                        }
                                        "INT32" -> {
                                            val intArray = IntArray(flatSize)
                                            tensor.intBuffer.get(intArray)
                                            intArray.map { it.toLong() }
                                        }
                                        else -> {
                                            result.error("CONVERSION_ERROR", "Cannot convert to int64", null)
                                            return
                                        }
                                    }
                                }
                            }
                            "uint8" -> {
                                try {
                                    val byteArray = ByteArray(flatSize)
                                    tensor.byteBuffer.get(byteArray)
                                    byteArray.map { it.toInt() and 0xFF }
                                } catch (e: Exception) {
                                    result.error("CONVERSION_ERROR", "Cannot convert to uint8", null)
                                    return
                                }
                            }
                            "bool" -> {
                                try {
                                    val byteArray = ByteArray(flatSize)
                                    tensor.byteBuffer.get(byteArray)
                                    byteArray.map { it != 0.toByte() }
                                } catch (e: Exception) {
                                    result.error("CONVERSION_ERROR", "Cannot convert to bool", null)
                                    return
                                }
                            }
                            else -> {
                                result.error("UNSUPPORTED_TYPE", "Unsupported data type: $dataType", null)
                                return
                            }
                        }

                    // Return data with shape
                    val resultMap =
                        mapOf(
                            "data" to data,
                            "shape" to shape.toList(),
                        )

                    result.success(resultMap)
                } catch (e: Exception) {
                    result.error("DATA_EXTRACTION_ERROR", e.message, e.stackTraceToString())
                }
            }
            "releaseOrtValue" -> {
                try {
                    val valueId = call.argument<String>("valueId")

                    if (valueId == null) {
                        result.error("INVALID_ARGS", "Missing value ID", null)
                        return
                    }

                    val tensor = ortValues.remove(valueId)
                    if (tensor != null) {
                        try {
                            tensor.close()
                        } catch (e: Exception) {
                            // Log error but continue
                            Log.e("ORT_ERROR", "Error closing tensor: ${e.message}")
                        }
                    }

                    result.success(null)
                } catch (e: Exception) {
                    result.error("RELEASE_ERROR", e.message, e.stackTraceToString())
                }
            }
            else -> {
                result.notImplemented()
            }
        }
    }

    override fun onDetachedFromEngine(
        @NonNull binding: FlutterPlugin.FlutterPluginBinding,
    ) {
        // Close all OrtValues
        for (value in ortValues.values) {
            try {
                value.close()
            } catch (e: Exception) {
                // Ignore exceptions during cleanup
            }
        }
        ortValues.clear()

        // Close all sessions
        for (session in sessions.values) {
            try {
                session.close()
            } catch (e: Exception) {
                // Ignore exceptions during cleanup
            }
        }
        sessions.clear()

        // Close the environment
        try {
            ortEnvironment.close()
        } catch (e: Exception) {
            // Ignore exceptions during cleanup
        }

        channel.setMethodCallHandler(null)
    }
}
