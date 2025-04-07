// ignore_for_file: use_key_in_widget_constructors, library_private_types_in_public_api, prefer_final_fields

import 'package:flutter/material.dart';
import 'painters/drawing_painter.dart';
import 'services/model_service.dart';
import 'dart:math' as math;

class DrawingPage extends StatefulWidget {
  @override
  _DrawingPageState createState() => _DrawingPageState();
}

class _DrawingPageState extends State<DrawingPage> {
  List<List<Offset>> _strokes = [];
  List<Offset> _currentStroke = [];
  String result = 'Chưa vẽ';
  double confidence = 0.0;
  Rect? boundingBox;
  late ModelService _modelService;

  @override
  void initState() {
    super.initState();
    _modelService = ModelService();
    _initializeModel();
  }

  Future<void> _initializeModel() async {
    try {
      await _modelService.loadModel();
      await _modelService.loadClasses(context);
    } catch (e) {
      setState(() {
        result = e.toString();
      });
    }
  }

  void _updateBoundingBox(Offset point) {
    if (boundingBox == null) {
      boundingBox = Rect.fromPoints(point, point);
    } else {
      boundingBox = Rect.fromPoints(
        Offset(
          math.min(boundingBox!.left, point.dx),
          math.min(boundingBox!.top, point.dy),
        ),
        Offset(
          math.max(boundingBox!.right, point.dx),
          math.max(boundingBox!.bottom, point.dy),
        ),
      );
    }
  }

  Future<void> _predict() async {
    if (_strokes.isEmpty) {
      setState(() {
        result = 'Chưa vẽ';
        confidence = 0.0;
      });
      return;
    }

    try {
      final (predictedClass, predictedConfidence) =
          await _modelService.predict(_strokes, boundingBox);

      setState(() {
        result = predictedClass;
        confidence = predictedConfidence;
        _strokes.clear();
        _currentStroke = [];
        boundingBox = null;
      });
    } catch (e) {
      setState(() {
        result = 'Lỗi dự đoán';
        confidence = 0.0;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Quick Draw App'),
      ),
      body: Column(
        children: [
          Expanded(
            child: Container(
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
              ),
              child: GestureDetector(
                onPanStart: (details) {
                  setState(() {
                    _currentStroke = [details.localPosition];
                    _updateBoundingBox(details.localPosition);
                  });
                },
                onPanUpdate: (details) {
                  setState(() {
                    _currentStroke.add(details.localPosition);
                    _updateBoundingBox(details.localPosition);
                  });
                },
                onPanEnd: (details) {
                  setState(() {
                    if (_currentStroke.isNotEmpty) {
                      _strokes.add(List.from(_currentStroke));
                      _currentStroke = [];
                    }
                  });
                },
                child: CustomPaint(
                  painter:
                      DrawingPainter(_strokes, _currentStroke, boundingBox),
                  size: Size.infinite,
                ),
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Text(
              "Result: $result (Confidence: ${confidence.toStringAsFixed(2)})",
              style: TextStyle(fontSize: 20),
            ),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton(onPressed: _predict, child: Text('Predict')),
              ElevatedButton(
                  onPressed: () => setState(() {
                        _strokes.clear();
                        _currentStroke = [];
                        boundingBox = null;
                        result = 'Chưa vẽ';
                        confidence = 0.0;
                      }),
                  child: Text('Clear')),
            ],
          ),
        ],
      ),
    );
  }
}
