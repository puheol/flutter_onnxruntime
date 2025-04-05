package com.masicai.flutter_onnxruntime

import android.util.Log
import androidx.annotation.NonNull
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtException
import java.nio.ByteBuffer
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.io.File
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OnnxTensor
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer

/** FlutterOnnxruntimePlugin */
class FlutterOnnxruntimePlugin: FlutterPlugin, MethodCallHandler {
  /// The MethodChannel that will the communication between Flutter and native Android
  ///
  /// This local reference serves to register the plugin with the Flutter Engine and unregister it
  /// when the Flutter Engine is detached from the Activity
  private lateinit var channel : MethodChannel
  private lateinit var ortEnvironment: OrtEnvironment
  private val sessions = ConcurrentHashMap<String, OrtSession>()

  override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
    channel = MethodChannel(flutterPluginBinding.binaryMessenger, "flutter_onnxruntime")
    channel.setMethodCallHandler(this)
    ortEnvironment = OrtEnvironment.getEnvironment()
  }

  override fun onMethodCall(@NonNull call: MethodCall, @NonNull result: Result) {
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
          
          result.success(mapOf(
            "sessionId" to sessionId,
            "inputNames" to inputNames,
            "outputNames" to outputNames
          ))
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
              val shape = if (inputs.containsKey(shapeName) && inputs[shapeName] is List<*>) {
                (inputs[shapeName] as List<*>).map { (it as Number).toLong() }.toLongArray()
              } else {
                // Default shape for 1D data
                when (value) {
                  is List<*> -> longArrayOf(value.size.toLong())
                  is ByteArray -> longArrayOf(value.size.toLong())
                  is FloatArray -> longArrayOf(value.size.toLong())
                  is IntArray -> longArrayOf(value.size.toLong())
                  else -> {
                    result.error("MISSING_SHAPE", 
                      "Shape information required for input '$name' of type: ${value?.javaClass?.name}", null)
                    return
                  }
                }
              }
              
              // Create appropriate tensor based on input type
              when (value) {
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
                      result.error("UNSUPPORTED_INPUT_TYPE", 
                        "Unsupported input element type for '$name': ${firstElement?.javaClass?.name}", null)
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
                  result.error("UNSUPPORTED_INPUT_TYPE", 
                    "Unsupported input type for '$name': ${value?.javaClass?.name}", null)
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
              val actualTensor = when {
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
      else -> {
        result.notImplemented()
      }
    }
  }

  override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
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
