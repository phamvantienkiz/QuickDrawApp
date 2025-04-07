import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:math' as math;

class ModelService {
  Interpreter? _interpreter;
  List<String> classes = [];

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
          'assets/quickdraw_cnn_optimized_best.tflite');
      print('Model loaded successfully');
      final inputShape = _interpreter!.getInputTensor(0).shape;
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      print('Input shape: $inputShape');
      print('Output shape: $outputShape');
    } catch (e) {
      print('Error loading model: $e');
      throw Exception('Lỗi tải mô hình');
    }
  }

  Future<void> loadClasses(BuildContext context) async {
    try {
      final String classData =
          await DefaultAssetBundle.of(context).loadString('assets/classes.txt');
      classes = classData
          .split('\n')
          .where((line) => line.trim().isNotEmpty)
          .toList();
      print('Classes loaded: $classes');
    } catch (e) {
      print('Error loading classes: $e');
      throw Exception('Lỗi tải danh sách lớp');
    }
  }

  Future<(String, double)> predict(
      List<List<Offset>> strokes, Rect? boundingBox) async {
    if (strokes.isEmpty || _interpreter == null || classes.isEmpty) {
      return ('Chưa vẽ', 0.0);
    }

    final List<List<double>> normalizedPoints =
        _normalizePoints(strokes, boundingBox);
    final imgData = await _drawToImage(normalizedPoints);

    if (imgData == null) {
      print('Failed to get image data');
      return ('Lỗi xử lý ảnh', 0.0);
    }

    try {
      final processedImage = _processImage(imgData);
      final prediction = await _runInference(processedImage);
      return prediction;
    } catch (e) {
      print('Error running inference: $e');
      return ('Lỗi dự đoán', 0.0);
    }
  }

  List<List<double>> _normalizePoints(
      List<List<Offset>> strokes, Rect? boundingBox) {
    if (boundingBox == null) return [];

    final double scale = 20.0 / math.max(boundingBox.width, boundingBox.height);
    final double xOffset =
        (28 - boundingBox.width * scale) / 2 - boundingBox.left * scale;
    final double yOffset =
        (28 - boundingBox.height * scale) / 2 - boundingBox.top * scale;

    List<List<double>> normalizedStrokes = [];

    for (var stroke in strokes) {
      if (stroke.length <= 1) continue;

      List<double> normalizedStroke = [];
      for (var point in stroke) {
        normalizedStroke.add(point.dx * scale + xOffset);
        normalizedStroke.add(point.dy * scale + yOffset);
      }

      normalizedStrokes.add(normalizedStroke);
    }

    return normalizedStrokes;
  }

  Future<ByteData?> _drawToImage(List<List<double>> normalizedPoints) async {
    final imgWidth = 28;
    final imgHeight = 28;
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(
        recorder,
        Rect.fromPoints(const Offset(0, 0),
            Offset(imgWidth.toDouble(), imgHeight.toDouble())));

    canvas.drawColor(Colors.white, BlendMode.src);

    final paint = Paint()
      ..color = Colors.black
      ..strokeWidth = 2.0
      ..strokeCap = StrokeCap.round;

    for (var normalizedStroke in normalizedPoints) {
      if (normalizedStroke.length < 4) continue;

      for (int i = 0; i < normalizedStroke.length - 3; i += 2) {
        canvas.drawLine(
          Offset(normalizedStroke[i], normalizedStroke[i + 1]),
          Offset(normalizedStroke[i + 2], normalizedStroke[i + 3]),
          paint,
        );
      }
    }

    final picture = recorder.endRecording();
    final img = await picture.toImage(imgWidth, imgHeight);
    return img.toByteData(format: ui.ImageByteFormat.rawRgba);
  }

  Float32List _processImage(ByteData imgData) {
    final rgba = imgData.buffer.asUint8List();
    final imgWidth = 28;
    final imgHeight = 28;
    var grayscaleImage = List<List<double>>.generate(
        imgHeight, (_) => List<double>.filled(imgWidth, 0));

    for (int y = 0; y < imgHeight; y++) {
      for (int x = 0; x < imgWidth; x++) {
        final pixelIdx = (y * imgWidth + x) * 4;
        final r = rgba[pixelIdx];
        final g = rgba[pixelIdx + 1];
        final b = rgba[pixelIdx + 2];
        final a = rgba[pixelIdx + 3];

        if ((x == 0 && y == 0) || (x == 27 && y == 27)) {
          print('Pixel ($x,$y): R=$r, G=$g, B=$b, A=$a');
        }

        double gray = 0;
        if (a > 0) {
          gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
          gray = 1.0 - gray;
          gray = (gray > 0.5) ? 1.0 : 0.0;
        }

        grayscaleImage[y][x] = gray;
      }
    }

    var flatImage = Float32List(1 * imgHeight * imgWidth * 1);
    int index = 0;
    for (int y = 0; y < imgHeight; y++) {
      for (int x = 0; x < imgWidth; x++) {
        flatImage[index++] = grayscaleImage[y][x];
      }
    }

    return flatImage;
  }

  Future<(String, double)> _runInference(Float32List processedImage) async {
    final outputShape = [1, classes.length];
    final output = List<List<double>>.generate(
        outputShape[0], (_) => List<double>.filled(outputShape[1], 0));

    print('Input shape: [1, 28, 28, 1]');
    print('Output shape: $outputShape');

    _interpreter!.resizeInputTensor(0, [1, 28, 28, 1]);
    _interpreter!.allocateTensors();

    _interpreter!.run(processedImage.buffer, output);

    final resultArray = output[0];
    print('Output values: $resultArray');

    double maxProb = -1;
    int maxIndex = -1;
    for (int i = 0; i < resultArray.length; i++) {
      if (resultArray[i] > maxProb) {
        maxProb = resultArray[i];
        maxIndex = i;
      }
    }

    if (maxIndex >= 0 && maxIndex < classes.length) {
      return (classes[maxIndex], maxProb);
    } else {
      return ('Unknown', 0.0);
    }
  }
}
