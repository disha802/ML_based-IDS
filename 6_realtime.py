"""
REAL-TIME IDS DEPLOYMENT (HYBRID MODEL)
Deploys the best hybrid-trained model for real-time intrusion detection
Optimized for performance and accuracy
"""

import pyshark
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ===============================
# CONFIGURATION
# ===============================
INTERFACE = "7"  # Change to your network interface
FLOW_TIMEOUT = 2.0  # Flow timeout in seconds
BUFFER_SIZE = 50  # Process every N packets
ALERT_THRESHOLD = 0.70  # 70% confidence threshold

# Model files (hybrid trained)
MODEL_FILE = "BEST_FINAL_model.pkl"
SCALER_FILE = "FINAL_scaler.pkl"
LABEL_ENCODER_FILE = "FINAL_label_encoder.pkl"
FEATURES_FILE = "FINAL_feature_names.txt"

print("="*80)
print("🛡️  REAL-TIME INTRUSION DETECTION SYSTEM")
print("   Hybrid Model Deployment")
print("="*80)

# ===============================
# FLOW AGGREGATOR
# ===============================

class FlowAggregator:
    """Aggregate packets into flows with timeout"""
    
    def __init__(self, timeout=2.0):
        self.timeout = timeout
        self.flows = {}
        self.last_cleanup = time.time()
    
    def _create_flow_key(self, packet_info):
        """Create bidirectional flow key"""
        # Sort IPs to make bidirectional
        ip1, ip2 = sorted([packet_info['src_ip'], packet_info['dst_ip']])
        return f"{ip1}_{ip2}_{packet_info['protocol']}"
    
    def add_packet(self, packet_info):
        """Add packet to flow"""
        flow_key = self._create_flow_key(packet_info)
        timestamp = packet_info['timestamp']
        
        if flow_key not in self.flows:
            self.flows[flow_key] = {
                'protocol': packet_info['protocol'],
                'src_ip': packet_info['src_ip'],
                'dst_ip': packet_info['dst_ip'],
                'packets': [],
                'timestamps': [],
                'lengths': [],
                'start_time': timestamp,
                'last_update': timestamp
            }
        
        # Update flow
        self.flows[flow_key]['packets'].append(packet_info)
        self.flows[flow_key]['timestamps'].append(timestamp)
        self.flows[flow_key]['lengths'].append(packet_info['packet_length'])
        self.flows[flow_key]['last_update'] = timestamp
    
    def extract_flow_features(self, flow_data):
        """Extract features matching training"""
        timestamps = flow_data['timestamps']
        lengths = flow_data['lengths']
        
        # Calculate duration
        if len(timestamps) < 2:
            duration = 0.0
        else:
            duration = max(timestamps) - min(timestamps)
        
        # Match training features
        features = {
            'protocol': flow_data['protocol'],
            'flow_duration': duration,
            'total_packets': len(timestamps),
            'total_bytes': sum(lengths),
            'packets_per_second': len(timestamps) / max(duration, 0.001),
            'bytes_per_second': sum(lengths) / max(duration, 0.001),
        }
        
        return features
    
    def get_active_flows(self, current_time):
        """Get flows ready for classification"""
        active = []
        to_remove = []
        
        for flow_key, flow_data in list(self.flows.items()):
            time_since_update = current_time - flow_data['last_update']
            
            # Flow completed (timeout) or has enough packets
            if time_since_update >= self.timeout or len(flow_data['packets']) >= 10:
                if len(flow_data['packets']) >= 2:  # At least 2 packets
                    features = self.extract_flow_features(flow_data)
                    features['flow_key'] = flow_key
                    active.append(features)
                to_remove.append(flow_key)
        
        # Clean up completed flows
        for flow_key in to_remove:
            del self.flows[flow_key]
        
        return active
    
    def cleanup_old_flows(self, current_time):
        """Remove very old flows"""
        if current_time - self.last_cleanup < 10:
            return
        
        to_remove = []
        for flow_key, flow_data in list(self.flows.items()):
            if current_time - flow_data['start_time'] > 60:  # 1 minute max
                to_remove.append(flow_key)
        
        for flow_key in to_remove:
            del self.flows[flow_key]
        
        self.last_cleanup = current_time

# ===============================
# REAL-TIME IDS
# ===============================

class RealTimeIDS:
    """Real-time intrusion detection system"""
    
    def __init__(self, model, scaler, feature_names, label_encoder):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.label_encoder = label_encoder
        self.aggregator = FlowAggregator(timeout=FLOW_TIMEOUT)
        self.detection_log = []
        self.stats = {
            'packets_processed': 0,
            'flows_classified': 0,
            'dos_detected': 0,
            'benign_detected': 0
        }
        
        print(f"✅ IDS initialized with features: {self.feature_names}")
    
    def process_packet(self, packet):
        """Process single packet"""
        try:
            if not hasattr(packet, 'ip'):
                return None
            
            packet_info = {
                'timestamp': float(packet.sniff_timestamp),
                'protocol': int(packet.ip.proto),
                'src_ip': packet.ip.src,
                'dst_ip': packet.ip.dst,
                'packet_length': int(packet.length)
            }
            
            self.aggregator.add_packet(packet_info)
            self.stats['packets_processed'] += 1
            return packet_info
            
        except Exception as e:
            return None
    
    def classify_flows(self, flows):
        """Classify flows using trained model"""
        if len(flows) == 0:
            return []
        
        # Convert to DataFrame
        flows_df = pd.DataFrame(flows)
        flow_keys = flows_df['flow_key'].tolist()
        flows_df = flows_df.drop('flow_key', axis=1)
        
        # Ensure correct feature order
        X = flows_df[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Process results
        results = []
        for i in range(len(flows_df)):
            label = self.label_encoder.inverse_transform([predictions[i]])[0]
            confidence = probabilities[i][predictions[i]]
            
            result = {
                'flow_key': flow_keys[i],
                'prediction': label,
                'confidence': confidence,
                'features': flows_df.iloc[i].to_dict(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            results.append(result)
            self.stats['flows_classified'] += 1
            
            if label == 'DoS':
                self.stats['dos_detected'] += 1
            else:
                self.stats['benign_detected'] += 1
        
        return results
    
    def alert(self, detection):
        """Generate alert for DoS detection"""
        print(f"\n{'🚨 '*20}")
        print(f"⚠️  ATTACK DETECTED!")
        print(f"{'🚨 '*20}")
        print(f"Time:       {detection['timestamp']}")
        print(f"Flow:       {detection['flow_key']}")
        print(f"Prediction: {detection['prediction']}")
        print(f"Confidence: {detection['confidence']:.2%}")
        print(f"Flow Stats:")
        print(f"  Packets: {detection['features']['total_packets']}")
        print(f"  Bytes:   {detection['features']['total_bytes']}")
        print(f"  PPS:     {detection['features']['packets_per_second']:.1f}")
        print(f"  BPS:     {detection['features']['bytes_per_second']:.1f}")
        print(f"{'🚨 '*20}\n")
        
        self.detection_log.append(detection)
    
    def check_flows(self, current_time):
        """Check and classify active flows"""
        # Get flows ready for classification
        active_flows = self.aggregator.get_active_flows(current_time)
        
        if len(active_flows) == 0:
            return
        
        # Classify flows
        detections = self.classify_flows(active_flows)
        
        # Alert on attacks
        for detection in detections:
            if detection['prediction'] == 'DoS':
                if detection['confidence'] >= ALERT_THRESHOLD:
                    self.alert(detection)
        
        # Cleanup old flows
        self.aggregator.cleanup_old_flows(current_time)
    
    def print_stats(self):
        """Print current statistics"""
        print(f"📊 Statistics:")
        print(f"   Packets:  {self.stats['packets_processed']:6d}")
        print(f"   Flows:    {self.stats['flows_classified']:6d}")
        print(f"   DoS:      {self.stats['dos_detected']:6d}")
        print(f"   Benign:   {self.stats['benign_detected']:6d}")
        print(f"   Alerts:   {len(self.detection_log):6d}")
    
    def save_log(self, filename="hybrid_detection_log.csv"):
        """Save detection log"""
        if len(self.detection_log) > 0:
            df = pd.DataFrame(self.detection_log)
            df.to_csv(filename, index=False)
            print(f"✅ Detection log saved: {filename}")

# ===============================
# MAIN DEPLOYMENT
# ===============================

def deploy_realtime():
    """Deploy real-time IDS"""
    
    print("\n" + "="*80)
    print("LOADING TRAINED MODEL")
    print("="*80)
    
    # Load model
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded: {MODEL_FILE}")
        
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler loaded: {SCALER_FILE}")
        
        with open(FEATURES_FILE, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"✅ Features loaded: {feature_names}")
        
        with open(LABEL_ENCODER_FILE, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"✅ Label encoder loaded")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found!")
        print(f"   {e}")
        print(f"\n💡 Run HYBRID_training.py first to generate models.")
        return
    
    # Initialize IDS
    ids = RealTimeIDS(model, scaler, feature_names, label_encoder)
    
    print("\n" + "="*80)
    print("STARTING PACKET CAPTURE")
    print("="*80)
    print(f"   Interface: {INTERFACE}")
    print(f"   Alert threshold: {ALERT_THRESHOLD:.0%}")
    print(f"   Flow timeout: {FLOW_TIMEOUT}s")
    print(f"   Buffer size: {BUFFER_SIZE} packets")
    print("\n💡 Press Ctrl+C to stop...\n")
    
    # Start capture
    capture = pyshark.LiveCapture(interface=INTERFACE)
    
    packet_count = 0
    start_time = time.time()
    last_stats = time.time()
    
    try:
        for packet in capture.sniff_continuously():
            # Process packet
            packet_info = ids.process_packet(packet)
            
            if packet_info is None:
                continue
            
            packet_count += 1
            
            # Check flows periodically
            if packet_count % BUFFER_SIZE == 0:
                current_time = time.time()
                ids.check_flows(current_time)
                
                # Print stats every 10 seconds
                if current_time - last_stats >= 10:
                    elapsed = current_time - start_time
                    pps = packet_count / elapsed
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
                    print(f"   PPS: {pps:6.1f}")
                    ids.print_stats()
                    last_stats = current_time
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping IDS...")
        
        # Final check
        ids.check_flows(time.time())
        
        # Save log
        ids.save_log()
        
        # Summary
        print("\n" + "="*80)
        print("SESSION SUMMARY")
        print("="*80)
        elapsed = time.time() - start_time
        print(f"Duration:    {elapsed:.1f} seconds")
        ids.print_stats()
        print(f"Avg PPS:     {packet_count/elapsed:.1f}")
        
        if len(ids.detection_log) > 0:
            print(f"\n🚨 Total alerts: {len(ids.detection_log)}")
            dos_conf = [d['confidence'] for d in ids.detection_log if d['prediction']=='DoS']
            if dos_conf:
                print(f"   Avg confidence: {np.mean(dos_conf):.2%}")
                print(f"   Max confidence: {np.max(dos_conf):.2%}")
        
        print("\n✅ IDS stopped successfully")
    
    finally:
        capture.close()

if __name__ == "__main__":
    deploy_realtime()
    