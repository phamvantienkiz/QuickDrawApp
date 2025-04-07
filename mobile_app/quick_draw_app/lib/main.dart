import 'package:flutter/material.dart';
import 'drawing_page.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Quick Draw',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: DrawingPage(),
    );
  }
}
