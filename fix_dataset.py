"""
Auto-Find and Fix FER2013 Dataset File
Looks for the dataset file and renames it properly
"""

import os
import shutil

def find_and_fix_fer2013():
    """Find the FER2013 file and rename it if needed"""
    
    print("=" * 70)
    print("FER2013 AUTO-FIX SCRIPT")
    print("=" * 70)
    
    current_dir = os.getcwd()
    print(f"\n📍 Looking in: {current_dir}\n")
    
    # Check if fer2013.csv already exists
    if os.path.exists('fer2013.csv'):
        size = os.path.getsize('fer2013.csv') / (1024 * 1024)
        print(f"\n✅ fer2013.csv already exists! ({size:.2f} MB)")
        if size > 50:
            print("   File looks good!")
            return True
        else:
            print("   ⚠️  File seems too small...")
    
    # List all files and folders
    print("📂 Contents of current directory:")
    for item in os.listdir('.'):
        if os.path.isfile(item):
            size = os.path.getsize(item) / (1024 * 1024)
            print(f"   📄 {item:40s} ({size:.2f} MB)")
        else:
            print(f"   📁 {item}/")
    
    # Look for likely candidates in current directory AND subdirectories
    print("\n🔍 Searching for FER2013 dataset file (including subdirectories)...\n")
    
    candidates = []
    
    # Search in current directory
    for filename in os.listdir('.'):
        filepath = os.path.join('.', filename)
        if not os.path.isfile(filepath):
            continue
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        
        # Check if it's likely the FER2013 file
        if 80 < file_size < 120:  # Around 90-100 MB
            candidates.append((filepath, filename, file_size))
            print(f"   Found candidate: {filename} ({file_size:.2f} MB)")
    
    # Search in subdirectories (one level deep)
    for item in os.listdir('.'):
        if os.path.isdir(item):
            print(f"\n   🔍 Searching inside folder: {item}/")
            try:
                for subfile in os.listdir(item):
                    subpath = os.path.join(item, subfile)
                    if os.path.isfile(subpath):
                        file_size = os.path.getsize(subpath) / (1024 * 1024)
                        print(f"      📄 {subfile:35s} ({file_size:.2f} MB)")
                        
                        # Check for fer2013.csv specifically or large CSV files
                        if subfile.lower() == 'fer2013.csv' or (80 < file_size < 120):
                            candidates.append((subpath, subfile, file_size))
                            print(f"      ⭐ CANDIDATE FOUND!")
            except PermissionError:
                print(f"      ⚠️  No permission to access")
                continue
    
    if not candidates:
        print("\n❌ No candidate files found!")
        print("\n💡 Things to check:")
        print("   1. Did you extract the ZIP file?")
        print("   2. Look inside the extracted folder for fer2013.csv")
        print("   3. File should be around 90-100 MB")
        print("\n📥 Download location:")
        print("   https://www.kaggle.com/datasets/msambare/fer2013")
        return False
    
    # If we found candidates, offer to copy/rename
    print(f"\n💡 Found {len(candidates)} potential file(s)")
    
    for i, (filepath, filename, size) in enumerate(candidates, 1):
        if filename == 'fer2013.csv' and filepath == './fer2013.csv':
            continue
        
        # Try to read first line to confirm it's CSV
        try:
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
            
            if 'emotion' in first_line and 'pixels' in first_line:
                print(f"\n✅ This looks like the FER2013 dataset!")
                print(f"   Location: {filepath}")
                print(f"   Filename: {filename}")
                print(f"   Size: {size:.2f} MB")
                print(f"   First line: {first_line[:60]}...")
                
                response = input(f"\n   Copy this file to './fer2013.csv'? (y/n): ").lower()
                
                if response == 'y':
                    try:
                        shutil.copy(filepath, 'fer2013.csv')
                        print(f"\n✅ SUCCESS! Created fer2013.csv in current directory")
                        print(f"   Original file kept at: {filepath}")
                        return True
                    except Exception as e:
                        print(f"\n❌ Error: {e}")
                        print("\n💡 Try manually copying the file:")
                        print(f"   1. Open the folder containing: {filepath}")
                        print(f"   2. Copy '{filename}'")
                        print(f"   3. Paste it in: {current_dir}")
                        print(f"   4. Rename to: fer2013.csv")
                        return False
        except:
            print(f"   ⚠️  Could not read {filename} as text")
    
    print("\n💡 Manual steps:")
    print("   1. Find the file that's ~90-100 MB")
    print("   2. Right-click → Rename")
    print("   3. Name it: fer2013.csv")
    print("   4. Run: python verify_dataset.py")
    
    return False

if __name__ == "__main__":
    find_and_fix_fer2013()
    
    print("\n" + "=" * 70)
    print("\n🔄 Next step: Run 'python verify_dataset.py' to confirm!")