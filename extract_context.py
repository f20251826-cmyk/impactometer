import os
import json
import glob
from layer4_intervention import build_context_payload

def main():
    # Find the latest output analysis file in output/
    files = glob.glob(os.path.join("output", "analysis_basel_*.json"))
    if not files:
        print("❌ No session analysis output files found in output/")
        return
    
    # Get the latest file by modification time
    latest_file = max(files, key=os.path.getmtime)
    print(f"📂 Found latest analysis file: {latest_file}")
    
    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract components
    layer1_trans = data.get("layer1_transcription", {})
    transcript_data = {
        "transcript": layer1_trans.get("transcript", ""),
        "speakers": layer1_trans.get("speakers", [])
    }
    
    breakdown_data = data.get("layer2_breakdown", {})
    classification_data = data.get("layer3_classification", {})
    
    # Reconstruct layer1_metadata
    speakers_list = transcript_data["speakers"]
    confidences = [s.get("confidence", 0.99) for s in speakers_list]
    layer1_metadata = {
        "speaker_count": 4, # Basel has Gaurav Shah, Ram, Vihaan, and unidentified speaker
        "duration": 2076.0,
        "confidence_scores": confidences if confidences else [0.99] * 50
    }
    
    user_context = {
        "summary_length": "standard",
        "focus_areas": ["GTM", "ICP", "pitch deck", "focus"],
        "custom_notes": "Mentor-founder session between Gaurav Shah (PE/VC Advisor) and Ram Ramarathnam (Founder of Basel - DC appliances/green energy)."
    }
    
    # Build the payload
    context_payload = build_context_payload(
        transcript_data=transcript_data,
        breakdown_data=breakdown_data,
        classification_data=classification_data,
        layer1_metadata=layer1_metadata,
        user_context=user_context
    )
    
    # Save to context.json
    output_path = "context.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(context_payload, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Context file successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
