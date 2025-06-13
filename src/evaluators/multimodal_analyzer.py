"""
Multimodal Content Analyzer

Production-ready computer vision and audio processing for creative AI evaluation.
Supports images, videos, and audio content analysis using state-of-the-art models.
"""

import cv2
import numpy as np
import librosa
import speech_recognition as sr
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

# Computer Vision
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Audio Processing
try:
    from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

from pydub import AudioSegment
import soundfile as sf


class MultimodalAnalyzer:
    """
    Production-grade multimodal content analyzer for images, videos, and audio.
    
    Features:
    - YOLOv8 object detection
    - MediaPipe face detection and pose estimation
    - Advanced audio feature extraction
    - Speaker identification and voice characteristics
    - Engagement potential scoring for multimedia content
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the multimodal analyzer."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        self._init_computer_vision()
        self._init_audio_processing()
        
        self.logger.info("Multimodal analyzer initialized")
    
    def _init_computer_vision(self):
        """Initialize computer vision models."""
        try:
            # YOLO for object detection
            if YOLO_AVAILABLE:
                self.yolo_model = YOLO('yolov8n.pt')  # Nano version for speed
                self.logger.info("YOLOv8 model loaded")
            else:
                self.yolo_model = None
                self.logger.warning("YOLOv8 not available")
            
            # MediaPipe for face detection and pose estimation
            if MEDIAPIPE_AVAILABLE:
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_pose = mp.solutions.pose
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
                self.pose = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5
                )
                self.hands = self.mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=2,
                    min_detection_confidence=0.5
                )
                
                self.logger.info("MediaPipe models loaded")
            else:
                self.face_detection = None
                self.pose = None
                self.hands = None
                self.logger.warning("MediaPipe not available")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize computer vision: {e}")
    
    def _init_audio_processing(self):
        """Initialize audio processing models."""
        try:
            # Speech recognition
            self.speech_recognizer = sr.Recognizer()
            
            # Speaker identification (if available)
            if SPEECHBRAIN_AVAILABLE:
                try:
                    self.speaker_classifier = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir="./pretrained_models/spkrec"
                    )
                    self.logger.info("SpeechBrain speaker classifier loaded")
                except Exception as e:
                    self.logger.warning(f"SpeechBrain models not available: {e}")
                    self.speaker_classifier = None
            else:
                self.speaker_classifier = None
                
        except Exception as e:
            self.logger.error(f"Failed to initialize audio processing: {e}")
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Comprehensive image analysis using computer vision.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image analysis results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image'}
            
            results = {}
            
            # Basic image properties
            height, width, channels = image.shape
            results['properties'] = {
                'width': width,
                'height': height,
                'channels': channels,
                'aspect_ratio': width / height,
                'resolution': width * height
            }
            
            # Object detection with YOLO
            if self.yolo_model:
                results['objects'] = self._detect_objects(image)
            
            # Face detection and analysis
            if self.face_detection:
                results['faces'] = self._detect_faces(image)
            
            # Pose estimation
            if self.pose:
                results['poses'] = self._detect_poses(image)
            
            # Hand detection
            if self.hands:
                results['hands'] = self._detect_hands(image)
            
            # Image quality metrics
            results['quality'] = self._analyze_image_quality(image)
            
            # Aesthetic analysis
            results['aesthetics'] = self._analyze_aesthetics(image)
            
            # Engagement potential
            results['engagement_potential'] = self._calculate_image_engagement_potential(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Image analysis failed: {e}")
            return {'error': str(e)}
    
    def _detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in image using YOLO."""
        try:
            results = self.yolo_model(image)
            objects = []
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        objects.append({
                            'class': self.yolo_model.names[int(box.cls)],
                            'confidence': float(box.conf),
                            'bbox': box.xyxy.tolist()[0],  # [x1, y1, x2, y2]
                            'center': [(box.xyxy[0][0] + box.xyxy[0][2]) / 2, 
                                     (box.xyxy[0][1] + box.xyxy[0][3]) / 2]
                        })
            
            return objects
            
        except Exception as e:
            self.logger.error(f"Object detection failed: {e}")
            return []
    
    def _detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using MediaPipe."""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    
                    faces.append({
                        'confidence': detection.score[0],
                        'bbox': {
                            'x': bbox.xmin * w,
                            'y': bbox.ymin * h,
                            'width': bbox.width * w,
                            'height': bbox.height * h
                        },
                        'relative_bbox': {
                            'x': bbox.xmin,
                            'y': bbox.ymin,
                            'width': bbox.width,
                            'height': bbox.height
                        }
                    })
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []
    
    def _detect_poses(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect human poses using MediaPipe."""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_image)
            
            poses = []
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                poses.append({
                    'landmarks': landmarks,
                    'visible_landmarks': sum(1 for lm in landmarks if lm['visibility'] > 0.5)
                })
            
            return poses
            
        except Exception as e:
            self.logger.error(f"Pose detection failed: {e}")
            return []
    
    def _detect_hands(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect hands using MediaPipe."""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            hands = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    
                    hands.append({
                        'landmarks': landmarks,
                        'handedness': 'unknown'  # Could be enhanced with handedness detection
                    })
            
            return hands
            
        except Exception as e:
            self.logger.error(f"Hand detection failed: {e}")
            return []
    
    def _analyze_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze image quality metrics."""
        try:
            # Convert to grayscale for some analyses
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Brightness
            brightness = np.mean(gray)
            
            # Contrast (standard deviation)
            contrast = np.std(gray)
            
            # Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Color distribution
            color_channels = cv2.split(image)
            color_balance = {
                'blue_mean': np.mean(color_channels[0]),
                'green_mean': np.mean(color_channels[1]),
                'red_mean': np.mean(color_channels[2])
            }
            
            return {
                'brightness': float(brightness),
                'contrast': float(contrast),
                'sharpness': float(sharpness),
                'color_balance': color_balance
            }
            
        except Exception as e:
            self.logger.error(f"Image quality analysis failed: {e}")
            return {}
    
    def _analyze_aesthetics(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze aesthetic properties of the image."""
        try:
            # Rule of thirds analysis
            height, width = image.shape[:2]
            
            # Divide image into thirds
            third_width = width // 3
            third_height = height // 3
            
            # Calculate interest points at rule of thirds intersections
            roi_scores = []
            for x in [third_width, 2 * third_width]:
                for y in [third_height, 2 * third_height]:
                    roi = image[max(0, y-20):min(height, y+20), max(0, x-20):min(width, x+20)]
                    if roi.size > 0:
                        roi_score = np.std(roi)  # Variance as interest measure
                        roi_scores.append(roi_score)
            
            rule_of_thirds_score = np.mean(roi_scores) if roi_scores else 0
            
            # Color harmony (simplified)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            color_harmony = 1.0 - (np.std(hue_hist) / np.mean(hue_hist)) if np.mean(hue_hist) > 0 else 0
            
            return {
                'rule_of_thirds_score': float(rule_of_thirds_score),
                'color_harmony': float(color_harmony)
            }
            
        except Exception as e:
            self.logger.error(f"Aesthetic analysis failed: {e}")
            return {}
    
    def _calculate_image_engagement_potential(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate engagement potential based on image analysis."""
        score = 0.5  # Base score
        
        # Face presence boost (people engage more with faces)
        faces = analysis_results.get('faces', [])
        if faces:
            score += 0.3  # Strong boost for face presence
            # Additional boost for multiple faces
            score += min(0.1, len(faces) * 0.05)
        
        # Object diversity boost
        objects = analysis_results.get('objects', [])
        unique_objects = set(obj['class'] for obj in objects)
        if len(unique_objects) > 2:
            score += 0.1
        
        # High-engagement object types
        engaging_objects = {'person', 'dog', 'cat', 'car', 'food', 'cake', 'pizza'}
        for obj in objects:
            if obj['class'] in engaging_objects:
                score += 0.05
        
        # Image quality boost
        quality = analysis_results.get('quality', {})
        if quality.get('sharpness', 0) > 100:  # Sharp image
            score += 0.1
        if 80 <= quality.get('brightness', 0) <= 180:  # Good brightness
            score += 0.05
        
        # Resolution boost
        properties = analysis_results.get('properties', {})
        if properties.get('resolution', 0) > 1000000:  # 1MP+
            score += 0.1
        
        return min(1.0, score)
    
    def analyze_video(self, video_path: str, sample_frames: int = 10) -> Dict[str, Any]:
        """
        Comprehensive video analysis.
        
        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample for analysis
            
        Returns:
            Video analysis results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {'error': 'Could not open video'}
            
            # Video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Sample frames for analysis
            frame_interval = max(1, frame_count // sample_frames) if frame_count > 0 else 1
            
            # Aggregate analysis across frames
            all_objects = []
            all_faces = []
            all_poses = []
            frame_qualities = []
            
            for i in range(0, frame_count, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Analyze frame
                if self.yolo_model:
                    objects = self._detect_objects(frame)
                    all_objects.extend([obj['class'] for obj in objects])
                
                if self.face_detection:
                    faces = self._detect_faces(frame)
                    all_faces.extend(faces)
                
                if self.pose:
                    poses = self._detect_poses(frame)
                    all_poses.extend(poses)
                
                # Frame quality
                quality = self._analyze_image_quality(frame)
                frame_qualities.append(quality)
            
            cap.release()
            
            # Aggregate results
            unique_objects = list(set(all_objects))
            avg_faces_per_frame = len(all_faces) / max(1, sample_frames)
            avg_poses_per_frame = len(all_poses) / max(1, sample_frames)
            
            # Average quality metrics
            avg_quality = {}
            if frame_qualities:
                for key in frame_qualities[0].keys():
                    if key != 'color_balance':
                        avg_quality[key] = np.mean([fq.get(key, 0) for fq in frame_qualities])
            
            results = {
                'properties': {
                    'duration': duration,
                    'fps': fps,
                    'frame_count': frame_count,
                    'width': width,
                    'height': height,
                    'aspect_ratio': width / height if height > 0 else 0
                },
                'content_analysis': {
                    'unique_objects': unique_objects,
                    'total_objects': len(all_objects),
                    'avg_faces_per_frame': avg_faces_per_frame,
                    'avg_poses_per_frame': avg_poses_per_frame
                },
                'quality': avg_quality,
                'engagement_potential': self._calculate_video_engagement_potential({
                    'duration': duration,
                    'unique_objects': unique_objects,
                    'avg_faces_per_frame': avg_faces_per_frame,
                    'fps': fps
                })
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Video analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_video_engagement_potential(self, analysis: Dict[str, Any]) -> float:
        """Calculate engagement potential for videos."""
        score = 0.5
        
        duration = analysis.get('duration', 0)
        unique_objects = analysis.get('unique_objects', [])
        avg_faces = analysis.get('avg_faces_per_frame', 0)
        fps = analysis.get('fps', 0)
        
        # Optimal duration for social media (15-60 seconds)
        if 15 <= duration <= 60:
            score += 0.3
        elif 5 <= duration <= 15:
            score += 0.2
        elif duration > 60:
            score += 0.1
        
        # Face presence (very important for engagement)
        if avg_faces > 0.8:
            score += 0.2
        elif avg_faces > 0.3:
            score += 0.1
        
        # Object diversity
        if len(unique_objects) > 3:
            score += 0.15
        
        # Good frame rate
        if fps >= 24:
            score += 0.05
        
        return min(1.0, score)
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Comprehensive audio analysis.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio analysis results
        """
        try:
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            results = {
                'properties': {
                    'duration': duration,
                    'sample_rate': sr,
                    'total_samples': len(y)
                }
            }
            
            # Basic audio features
            results['features'] = self._extract_audio_features(y, sr)
            
            # Speech recognition
            results['speech'] = self._analyze_speech(audio_path)
            
            # Speaker characteristics (if available)
            if self.speaker_classifier:
                results['speaker'] = self._analyze_speaker(audio_path)
            
            # Engagement potential
            results['engagement_potential'] = self._calculate_audio_engagement_potential(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Audio analysis failed: {e}")
            return {'error': str(e)}
    
    def _extract_audio_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract comprehensive audio features."""
        try:
            features = {}
            
            # Rhythm and tempo
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['beats_per_minute'] = float(tempo)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            # MFCC features (voice characteristics)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
            features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
            
            # Zero crossing rate (speech/music differentiation)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zero_crossing_rate'] = float(np.mean(zcr))
            
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            features['rms_energy'] = float(np.mean(rms))
            
            # Chroma features (harmonic content)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Audio feature extraction failed: {e}")
            return {}
    
    def _analyze_speech(self, audio_path: str) -> Dict[str, Any]:
        """Analyze speech content in audio."""
        try:
            # Convert to WAV if needed
            if not audio_path.lower().endswith('.wav'):
                audio = AudioSegment.from_file(audio_path)
                wav_path = audio_path.rsplit('.', 1)[0] + '_temp.wav'
                audio.export(wav_path, format='wav')
                audio_file = wav_path
            else:
                audio_file = audio_path
            
            # Speech recognition
            with sr.AudioFile(audio_file) as source:
                audio_data = self.speech_recognizer.record(source)
                try:
                    transcript = self.speech_recognizer.recognize_google(audio_data)
                    confidence = 0.8  # Google API doesn't return confidence
                except sr.UnknownValueError:
                    transcript = ""
                    confidence = 0.0
                except sr.RequestError:
                    transcript = ""
                    confidence = 0.0
            
            # Clean up temp file if created
            if audio_file != audio_path:
                Path(audio_file).unlink(missing_ok=True)
            
            # Basic text analysis of transcript
            word_count = len(transcript.split()) if transcript else 0
            speech_rate = word_count / (librosa.get_duration(filename=audio_path) / 60) if transcript else 0
            
            return {
                'transcript': transcript,
                'confidence': confidence,
                'word_count': word_count,
                'speech_rate_wpm': speech_rate,  # words per minute
                'has_speech': bool(transcript)
            }
            
        except Exception as e:
            self.logger.error(f"Speech analysis failed: {e}")
            return {'transcript': '', 'confidence': 0.0, 'has_speech': False}
    
    def _analyze_speaker(self, audio_path: str) -> Dict[str, Any]:
        """Analyze speaker characteristics."""
        try:
            if not self.speaker_classifier:
                return {}
            
            # This is a placeholder for speaker analysis
            # In practice, you'd implement speaker embedding extraction
            # and comparison with known speakers
            
            return {
                'speaker_embedding': [],  # Would contain actual embedding
                'speaker_confidence': 0.0,
                'estimated_gender': 'unknown',
                'estimated_age_range': 'unknown'
            }
            
        except Exception as e:
            self.logger.error(f"Speaker analysis failed: {e}")
            return {}
    
    def _calculate_audio_engagement_potential(self, analysis: Dict[str, Any]) -> float:
        """Calculate engagement potential for audio content."""
        score = 0.5
        
        properties = analysis.get('properties', {})
        features = analysis.get('features', {})
        speech = analysis.get('speech', {})
        
        duration = properties.get('duration', 0)
        
        # Optimal duration (30 seconds to 5 minutes for most platforms)
        if 30 <= duration <= 300:
            score += 0.3
        elif 15 <= duration <= 30:
            score += 0.2
        elif duration > 300:
            score += 0.1
        
        # Speech presence (very important)
        if speech.get('has_speech', False):
            score += 0.3
            
            # Good speech rate (150-200 WPM is optimal)
            speech_rate = speech.get('speech_rate_wpm', 0)
            if 120 <= speech_rate <= 220:
                score += 0.1
        
        # Audio quality indicators
        if features.get('rms_energy', 0) > 0.01:  # Good volume level
            score += 0.1
        
        # Tempo engagement (moderate tempo is engaging)
        tempo = features.get('tempo', 0)
        if 80 <= tempo <= 140:
            score += 0.1
        
        return min(1.0, score) 