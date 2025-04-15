import 'package:flutter/material.dart';
import 'dart:async';

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final _flutterOnnxruntimePlugin = OnnxRuntime();
  final _aController = TextEditingController(); // Controller for the first input
  final _bController = TextEditingController(); // Controller for the second input
  String _result = ''; // Variable to store the result
  OrtSession? _session;

  @override
  void initState() {
    super.initState();
    _initializeSession();
  }

  Future<void> _initializeSession() async {
    _session = await _flutterOnnxruntimePlugin.createSessionFromAsset('assets/models/addition_model.ort');
    // print(_session?.inputNames);
    // print(_session?.outputNames);
  }

  @override
  void dispose() {
    _session?.close();
    _aController.dispose();
    _bController.dispose();
    super.dispose();
  }

  Future<void> _runAddition(double a, double b) async {
    if (_session == null) {
      await _initializeSession();
    }

    final inputA = await OrtValue.fromList([a], [1]);
    final inputB = await OrtValue.fromList([b], [1]);

    final inputs = {'A': inputA, 'B': inputB};

    final outputs = await _session!.run(inputs);

    // extract the output data
    try {
      final outputData = await outputs['C']!.asList();
      setState(() {
        _result = outputData[0].toString(); // Update the result with the output from the model
      });
    } catch (e) {
      setState(() {
        _result = 'Error: $e';
      });
    } finally {
      // clean up
      inputA.dispose();
      inputB.dispose();
      outputs['C']!.dispose();
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Addition Model with Onnxruntime Demo')),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(32.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                TextField(
                  controller: _aController,
                  keyboardType: TextInputType.number,
                  decoration: InputDecoration(labelText: 'Enter number A'),
                ),
                TextField(
                  controller: _bController,
                  keyboardType: TextInputType.number,
                  decoration: InputDecoration(labelText: 'Enter number B'),
                ),
                ElevatedButton(
                  onPressed: () {
                    final a = double.tryParse(_aController.text) ?? 0.0;
                    final b = double.tryParse(_bController.text) ?? 0.0;
                    _runAddition(a, b); // Run the addition when the button is pressed
                  },
                  child: Text('Add'),
                ),
                SizedBox(height: 20),
                Text('Result: $_result'), // Display the result
              ],
            ),
          ),
        ),
      ),
    );
  }
}
