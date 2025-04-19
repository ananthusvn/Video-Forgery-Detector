import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from datetime import datetime

class VideoForgeryDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize analysis variables
        self.prev_frame = None
        self.frame_diffs = []
        self.ssim_scores = []
        self.metadata = {}
        
    def extract_metadata(self):
        """Extract and analyze video metadata for inconsistencies"""
        # Get basic metadata
        self.metadata['duration'] = self.frame_count / self.fps
        self.metadata['resolution'] = f"{self.width}x{self.height}"
        self.metadata['fps'] = self.fps
        
        # Check for potential inconsistencies
        creation_time = datetime.fromtimestamp(os.path.getctime(self.video_path))
        modification_time = datetime.fromtimestamp(os.path.getmtime(self.video_path))
        self.metadata['creation_time'] = creation_time
        self.metadata['modification_time'] = modification_time
        
        # Flag if modification time is after creation time (potential re-encoding)
        if modification_time > creation_time:
            self.metadata['time_inconsistency'] = True
        else:
            self.metadata['time_inconsistency'] = False
            
        return self.metadata
    
    def analyze_frame_differences(self, sample_size=0.1):
        """Analyze frame-to-frame differences for abrupt changes"""
        sample_frames = max(1, int(self.frame_count * sample_size))
        step = max(1, self.frame_count // sample_frames)
        
        for i in range(0, self.frame_count, step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.prev_frame is not None:
                # Calculate absolute difference
                diff = cv2.absdiff(gray, self.prev_frame)
                diff_mean = np.mean(diff)
                self.frame_diffs.append(diff_mean)
                
                # Calculate structural similarity
                ssim_score = ssim(gray, self.prev_frame)
                self.ssim_scores.append(ssim_score)
                
            self.prev_frame = gray
            
        # Reset video capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return self.frame_diffs, self.ssim_scores
    
    def detect_double_compression(self):
        """Detect signs of double compression in video"""
        # This is a simplified approach - real implementation would be more complex
        quantization_artifacts = []
        
        sample_frames = min(10, self.frame_count)
        step = max(1, self.frame_count // sample_frames)
        
        for i in range(0, self.frame_count, step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Convert to YUV and analyze Y channel
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:,:,0]
            
            # DCT transform to detect quantization artifacts
            dct = cv2.dct(np.float32(y_channel)/255.0)
            dct_abs = np.abs(dct)
            
            # Look for blockiness in high frequencies
            blockiness = np.mean(dct_abs[8:, 8:])
            quantization_artifacts.append(blockiness)
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # High blockiness might indicate double compression
        avg_blockiness = np.mean(quantization_artifacts)
        threshold = 0.02  # Empirical threshold
        
        return avg_blockiness > threshold, avg_blockiness
    
    def detect_forgery(self):
        """Run all detection methods and return results"""
        results = {
            'metadata': self.extract_metadata(),
            'frame_analysis': {},
            'double_compression': {},
            'verdict': 'No significant signs of forgery detected',
            'confidence': 'Low'
        }
        
        # Frame difference analysis
        diffs, ssims = self.analyze_frame_differences()
        results['frame_analysis']['avg_frame_diff'] = np.mean(diffs)
        results['frame_analysis']['avg_ssim'] = np.mean(ssims)
        
        # Check for abrupt frame changes (potential frame insertion/deletion)
        diff_std = np.std(diffs)
        if diff_std > 10:  # Empirical threshold
            results['frame_analysis']['abrupt_changes'] = True
        else:
            results['frame_analysis']['abrupt_changes'] = False
            
        # Double compression detection
        double_comp, blockiness = self.detect_double_compression()
        results['double_compression']['detected'] = double_comp
        results['double_compression']['blockiness_score'] = blockiness
        
        # Make final assessment
        warning_flags = 0
        if results['metadata']['time_inconsistency']:
            warning_flags += 1
        if results['frame_analysis']['abrupt_changes']:
            warning_flags += 1
        if results['double_compression']['detected']:
            warning_flags += 1
            
        if warning_flags >= 2:
            results['verdict'] = 'Potential forgery detected'
            results['confidence'] = 'Medium'
        elif warning_flags == 1:
            results['verdict'] = 'Possible forgery indicators'
            results['confidence'] = 'Low'
            
        if warning_flags >= 1:
            results['verdict'] += '. Further forensic analysis recommended.'
            
        return results
    
    def visualize_analysis(self, results):
        """Create visualizations of the analysis"""
        plt.figure(figsize=(15, 10))
        
        # Frame differences plot
        plt.subplot(2, 2, 1)
        plt.plot(self.frame_diffs)
        plt.title('Frame-to-Frame Differences')
        plt.xlabel('Frame Index')
        plt.ylabel('Difference Magnitude')
        
        # SSIM plot
        plt.subplot(2, 2, 2)
        plt.plot(self.ssim_scores)
        plt.title('Structural Similarity Index')
        plt.xlabel('Frame Index')
        plt.ylabel('SSIM Score')
        
        # Metadata info
        plt.subplot(2, 2, 3)
        metadata_text = "\n".join([f"{k}: {v}" for k, v in results['metadata'].items()])
        plt.text(0.1, 0.5, metadata_text, fontsize=10)
        plt.axis('off')
        plt.title('Video Metadata')
        
        # Results summary
        plt.subplot(2, 2, 4)
        results_text = f"Verdict: {results['verdict']}\nConfidence: {results['confidence']}"
        results_text += f"\nFrame Analysis:\n  Avg Diff: {results['frame_analysis']['avg_frame_diff']:.2f}"
        results_text += f"\n  Avg SSIM: {results['frame_analysis']['avg_ssim']:.2f}"
        results_text += f"\nDouble Compression:\n  Detected: {results['double_compression']['detected']}"
        results_text += f"\n  Blockiness: {results['double_compression']['blockiness_score']:.4f}"
        plt.text(0.1, 0.5, results_text, fontsize=10)
        plt.axis('off')
        plt.title('Analysis Results')
        
        plt.tight_layout()
        plt.show()
        
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

def main():
    video_path = input("Enter path to video file: ")
    
    try:
        detector = VideoForgeryDetector(video_path)
        results = detector.detect_forgery()
        detector.visualize_analysis(results)
        
        print("\n=== Forgery Detection Results ===")
        print(f"Verdict: {results['verdict']}")
        print(f"Confidence: {results['confidence']}")
        print("\nDetails:")
        print(f"- Metadata inconsistencies: {results['metadata']['time_inconsistency']}")
        print(f"- Abrupt frame changes: {results['frame_analysis']['abrupt_changes']}")
        print(f"- Double compression detected: {results['double_compression']['detected']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()