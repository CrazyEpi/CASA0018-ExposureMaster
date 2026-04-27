import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/cupertino.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img_lib;
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter/services.dart';
import 'package:vibration/vibration.dart'; // Haptic actuator package

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    _cameras = await availableCameras();
  } catch (e) {
    print("Error fetching cameras: $e");
  }
  runApp(MaterialApp(
    theme: ThemeData.dark(),
    home: PolaroidExposureApp(),
  ));
}

class PolaroidExposureApp extends StatefulWidget {
  @override
  _PolaroidExposureAppState createState() => _PolaroidExposureAppState();
}

class _PolaroidExposureAppState extends State<PolaroidExposureApp> {
  CameraController? _controller; 
  Interpreter? _interpreter;
  
  String _result = "Awaiting permissions...";
  bool _isAnalyzing = false;
  bool _isCameraReady = false;

  // --- Photometric Exposure Parameters ---
  // Polaroid 600 Baseline: ISO 800, 1/30s, f/8
  final List<int> _isoValues = [100, 200, 400, 800, 1600, 3200];
  final List<double> _shutterValues = [1/250, 1/125, 1/60, 1/30, 1/15, 1/8];
  final List<String> _shutterLabels = ["1/250", "1/125", "1/60", "1/30", "1/15", "1/8"];
  final List<double> _apertureValues = [2.8, 4.0, 5.6, 8.0, 11.0, 16.0];

  // Current selected indices (Default to Baseline: 800, 1/30, 8.0)
  int _isoIdx = 3; 
  int _shutterIdx = 3; 
  int _apertureIdx = 3; 

  double _baselineEV = 0.0;
  double _currentEVOffset = 0.0;

  @override
  void initState() {
    super.initState();
    _calculateBaselineEV();
    _enforceHardwarePermissions(); 
    _initializeNeuralNetwork();
  }

  // Calculate the absolute EV100 for the Polaroid Baseline
  void _calculateBaselineEV() {
    // Formula: EV = log2(N^2 / t) - log2(ISO / 100)
    double n = _apertureValues[3]; // f/8
    double t = _shutterValues[3];  // 1/30
    double iso = _isoValues[3].toDouble(); // 800
    
    _baselineEV = (log(n * n / t) / ln2) - (log(iso / 100) / ln2);
  }

  // --- Hardware Permission Handling ---
  Future<void> _enforceHardwarePermissions() async {
    var status = await Permission.camera.status;
    if (status.isGranted) {
      _setupCameraSubsystem();
    } else {
      var requestResult = await Permission.camera.request();
      if (requestResult.isGranted) {
        _setupCameraSubsystem();
      } else {
        setState(() => _result = "Camera permission denied.");
      }
    }
  }

  // --- Neural Network Initialization ---
  Future<void> _initializeNeuralNetwork() async {
    try {
      // Extract model from assets to memory
      final rawAssetFile = await rootBundle.load('assets/exposure_expert_v4.tflite');
      final rawBytes = rawAssetFile.buffer.asUint8List();
      
      // Create interpreter from in-memory model buffer
      _interpreter = Interpreter.fromBuffer(rawBytes);
      
      setState(() => _result = "Model loaded. Awaiting capture.");
    } catch (e) {
      setState(() => _result = "Init Error: ${e.toString()}");
      print("Detailed Error: $e");
    }
  }

  // --- Camera Subsystem Setup ---
  Future<void> _setupCameraSubsystem() async {
    if (_cameras.isEmpty) return;
    _controller = CameraController(_cameras[0], ResolutionPreset.medium);
    try {
      await _controller!.initialize();
      await _controller!.setExposureMode(ExposureMode.locked);
      _applyExposureShift(); // Apply baseline offset upon startup
      if (mounted) {
        setState(() => _isCameraReady = true);
      }
    } catch (e) {
      print("Camera init error: $e");
    }
  }

  // --- Exposure Calculation Engine ---
  void _applyExposureShift() async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    double n = _apertureValues[_apertureIdx];
    double t = _shutterValues[_shutterIdx];
    double iso = _isoValues[_isoIdx].toDouble();

    // Calculate simulated EV for the selected settings
    double currentEV = (log(n * n / t) / ln2) - (log(iso / 100) / ln2);
    
    // EV Difference: If settings are brighter, offset must shift to balance
    double shift = _baselineEV - currentEV; 
    
    // Clamp to hardware-supported offset limits
    _currentEVOffset = shift.clamp(-2.0, 2.0);
    
    await _controller!.setExposureOffset(_currentEVOffset);
  }

  // --- Haptic Feedback Actuator ---
  void _triggerHapticFeedback(int resultIndex) async {
    bool? hasVibrator = await Vibration.hasVibrator();
    if (hasVibrator != true) return;

    if (resultIndex == 1) {
      // Optimal Exposure: Single short pulse (Success)
      Vibration.vibrate(duration: 100);
    } else {
      // Under/Over Exposure: Double heavy pulses (Warning)
      Vibration.vibrate(pattern: [0, 200, 100, 200]);
    }
  }

  // --- Inference Pipeline ---
  Future<void> _analyzeImage(XFile photo) async {
    if (_interpreter == null) return;
    setState(() { _isAnalyzing = true; _result = "Analyzing..."; });

    try {
      final bytes = await File(photo.path).readAsBytes();
      final image = img_lib.decodeImage(bytes);
      if (image == null) return;

      final resized = img_lib.copyResize(image, width: 224, height: 224);
      var input = List.generate(1, (i) => List.generate(224, (j) => List.generate(224, (k) => List.generate(3, (l) => 0.0))));
      
      for (var y = 0; y < 224; y++) {
        for (var x = 0; x < 224; x++) {
          final pixel = resized.getPixel(x, y);
          input[0][y][x][0] = pixel.r / 255.0;
          input[0][y][x][1] = pixel.g / 255.0;
          input[0][y][x][2] = pixel.b / 255.0;
        }
      }

      var output = List.generate(1, (i) => List.filled(3, 0.0));
      _interpreter!.run(input, output);

      final labels = ['Under-exposed', 'Optimal Exposure', 'Over-exposed'];
      int bestIdx = 0; 
      double maxProb = -1.0;
      for (int i = 0; i < 3; i++) {
        if (output[0][i] > maxProb) { 
          maxProb = output[0][i]; 
          bestIdx = i; 
        }
      }

      // Trigger the physical actuator (Vibration)
      _triggerHapticFeedback(bestIdx);

      setState(() {
        _result = "${labels[bestIdx]} (${(maxProb * 100).toStringAsFixed(1)}%)";
        _isAnalyzing = false;
      });
    } catch (e) {
      setState(() { 
        _result = "Inference Error: ${e.toString()}"; 
        _isAnalyzing = false; 
      });
    }
  }

  // --- UI Builder ---
  Widget _buildPicker(List<dynamic> items, int initialItem, ValueChanged<int> onChanged, String label) {
    return Expanded(
      child: Column(
        children: [
          Text(label, style: TextStyle(fontSize: 12, color: Colors.grey)),
          SizedBox(
            height: 80,
            child: CupertinoPicker(
              scrollController: FixedExtentScrollController(initialItem: initialItem),
              itemExtent: 32.0,
              onSelectedItemChanged: onChanged,
              children: items.map((e) => Center(child: Text(e.toString(), style: TextStyle(fontSize: 16)))).toList(),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return Scaffold(
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Text(
              _result, 
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.redAccent, fontSize: 16)
            ),
          )
        )
      );
    }

    return Scaffold(
      appBar: AppBar(title: Text("Polaroid Exposure Assistant")),
      body: Column(
        children: [
          Expanded(
            child: Center(
              child: AspectRatio(
                aspectRatio: 0.8,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    Padding(
                      padding: const EdgeInsets.all(20.0),
                      child: CameraPreview(_controller!),
                    ),
                    IgnorePointer(
                      child: Container(
                        decoration: BoxDecoration(
                          border: Border.all(color: Colors.white, width: 20),
                          borderRadius: BorderRadius.circular(10),
                        ),
                      ),
                    ),
                    if (_isAnalyzing) 
                      Container(
                        color: Colors.black45, 
                        child: Center(child: CircularProgressIndicator())
                      ),
                  ],
                ),
              ),
            ),
          ),
          
          Container(
            padding: EdgeInsets.fromLTRB(20, 10, 20, 20),
            color: Colors.black54,
            child: Column(
              children: [
                Text(
                  "Status: $_result", 
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 16, color: Colors.yellow)
                ),
                SizedBox(height: 4),
                Text(
                  "Simulated EV Shift: ${_currentEVOffset.toStringAsFixed(1)}", 
                  style: TextStyle(fontSize: 12, color: Colors.grey)
                ),
                Divider(),
                
                // Exposure Triangle Pickers
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    _buildPicker(_isoValues, _isoIdx, (val) {
                      setState(() => _isoIdx = val);
                      _applyExposureShift();
                    }, "ISO"),
                    _buildPicker(_shutterLabels, _shutterIdx, (val) {
                      setState(() => _shutterIdx = val);
                      _applyExposureShift();
                    }, "Shutter"),
                    _buildPicker(_apertureValues.map((e) => "f/$e").toList(), _apertureIdx, (val) {
                      setState(() => _apertureIdx = val);
                      _applyExposureShift();
                    }, "Aperture"),
                  ],
                ),
                
                SizedBox(height: 15),
                ElevatedButton(
                  onPressed: (_isAnalyzing || !_isCameraReady || _interpreter == null) ? null : () async {
                    try {
                      final photo = await _controller!.takePicture();
                      _analyzeImage(photo);
                    } catch (e) {
                      setState(() => _result = "Capture failed: ${e.toString()}");
                    }
                  },
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 15),
                    child: Text(_isAnalyzing ? "Processing..." : "Capture & Analyze"),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _controller?.dispose();
    _interpreter?.close();
    super.dispose();
  }
}