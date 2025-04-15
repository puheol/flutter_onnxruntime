// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_onnxruntime/src/ort_model_metadata.dart';

void main() {
  group('OrtModelMetadata', () {
    test('constructor initializes all fields correctly', () {
      final metadata = OrtModelMetadata(
        producerName: 'Test Producer',
        graphName: 'Test Graph',
        domain: 'test.domain',
        description: 'Test Description',
        version: 2,
        customMetadataMap: {'key1': 'value1', 'key2': 'value2'},
      );

      expect(metadata.producerName, 'Test Producer');
      expect(metadata.graphName, 'Test Graph');
      expect(metadata.domain, 'test.domain');
      expect(metadata.description, 'Test Description');
      expect(metadata.version, 2);
      expect(metadata.customMetadataMap, {'key1': 'value1', 'key2': 'value2'});
    });

    test('fromMap creates metadata with all fields from map', () {
      final map = {
        'producerName': 'Test Producer',
        'graphName': 'Test Graph',
        'domain': 'test.domain',
        'description': 'Test Description',
        'version': 2,
        'customMetadataMap': {'key1': 'value1', 'key2': 'value2'},
      };

      final metadata = OrtModelMetadata.fromMap(map);

      expect(metadata.producerName, 'Test Producer');
      expect(metadata.graphName, 'Test Graph');
      expect(metadata.domain, 'test.domain');
      expect(metadata.description, 'Test Description');
      expect(metadata.version, 2);
      expect(metadata.customMetadataMap, {'key1': 'value1', 'key2': 'value2'});
    });

    test('fromMap handles missing fields with default values', () {
      final map = <String, dynamic>{};

      final metadata = OrtModelMetadata.fromMap(map);

      expect(metadata.producerName, '');
      expect(metadata.graphName, '');
      expect(metadata.domain, '');
      expect(metadata.description, '');
      expect(metadata.version, 0);
      expect(metadata.customMetadataMap, isEmpty);
    });

    test('toMap converts all fields to map correctly', () {
      final metadata = OrtModelMetadata(
        producerName: 'Test Producer',
        graphName: 'Test Graph',
        domain: 'test.domain',
        description: 'Test Description',
        version: 2,
        customMetadataMap: {'key1': 'value1', 'key2': 'value2'},
      );

      final map = metadata.toMap();

      expect(map['producerName'], 'Test Producer');
      expect(map['graphName'], 'Test Graph');
      expect(map['domain'], 'test.domain');
      expect(map['description'], 'Test Description');
      expect(map['version'], 2);
      expect(map['customMetadataMap'], {'key1': 'value1', 'key2': 'value2'});
    });

    test('fromMap and toMap round trip preserves data', () {
      final original = OrtModelMetadata(
        producerName: 'Test Producer',
        graphName: 'Test Graph',
        domain: 'test.domain',
        description: 'Test Description',
        version: 2,
        customMetadataMap: {'key1': 'value1', 'key2': 'value2'},
      );

      final map = original.toMap();
      final roundTrip = OrtModelMetadata.fromMap(map);

      expect(roundTrip.producerName, original.producerName);
      expect(roundTrip.graphName, original.graphName);
      expect(roundTrip.domain, original.domain);
      expect(roundTrip.description, original.description);
      expect(roundTrip.version, original.version);
      expect(roundTrip.customMetadataMap, original.customMetadataMap);
    });
  });
}
