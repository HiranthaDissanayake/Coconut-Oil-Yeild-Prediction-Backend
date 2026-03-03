"""
╔══════════════════════════════════════════════════════════════════╗
║  BATCH DATA MANAGER - Switch Between Sample & Real Data         ║
║  Batch Data Manager - Sample සහ Real Data අතර Switch කරන්න    ║
╚══════════════════════════════════════════════════════════════════╝

This script helps you:
1. Train with sample/temporary data for testing
2. Switch to real data when ready
3. Retrain the oil model easily

මෙම script උදව් කරයි:
1. Testing සඳහා sample/temporary data train කිරීමට
2. Ready වූ විට real data ලදී switch කිරීමට
3. Oil model easily retrain කිරීමට
"""

import os
import shutil
import pandas as pd
from datetime import datetime

# ══════════════════════════════════════════════════════════════════
#  FILE PATHS
# ══════════════════════════════════════════════════════════════════
SAMPLE_CSV  = 'batch_data_SAMPLE.csv'     # Sample/temporary data
REAL_CSV    = 'batch_data_REAL.csv'        # Your actual measurements
ACTIVE_CSV  = 'batch_data.csv'             # Active file used by training script
BACKUP_DIR  = 'backups'                     # Backup folder

os.makedirs(BACKUP_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def show_menu():
    print("\n" + "═"*60)
    print("  BATCH DATA MANAGER / Batch Data Manager")
    print("═"*60)
    print("\nOptions / විකල්ප:")
    print("  1. Use SAMPLE data (for testing)")
    print("     SAMPLE data use කරන්න (testing සඳහා)")
    print()
    print("  2. Switch to REAL data (your actual measurements)")
    print("     REAL data ලදී switch කරන්න (ඔබේ actual measurements)")
    print()
    print("  3. Show current data (what's being used now)")
    print("     Current data show කරන්න (දැන් use වන දේ)")
    print()
    print("  4. Backup current data")
    print("     Current data backup කරන්න")
    print()
    print("  5. Exit / Exit")
    print()


def show_current_data():
    """Show which data is currently active"""
    print("\n" + "─"*60)
    print("  CURRENT DATA / Current Data")
    print("─"*60)
    
    if not os.path.exists(ACTIVE_CSV):
        print("\n  ⚠️  No active data file found!")
        print("  ⚠️  Active data file නැත!")
        print("\n  Run option 1 or 2 to set up data first.")
        print("  Data set කිරීමට option 1 හෝ 2 run කරන්න.")
        return
    
    df = pd.read_csv(ACTIVE_CSV)
    print(f"\n  File: {ACTIVE_CSV}")
    print(f"  Batches: {len(df)}")
    print(f"\n  Data:")
    print(df.to_string(index=False))
    
    # Check if it's sample or real data
    if 'Sample data for testing' in df['notes'].iloc[0]:
        print("\n  📊 Type: SAMPLE DATA (temporary)")
        print("  📊 Type: SAMPLE DATA (temporary)")
    else:
        print("\n  📊 Type: REAL DATA (your measurements)")
        print("  📊 Type: REAL DATA (ඔබේ measurements)")


def use_sample_data():
    """Switch to sample data"""
    print("\n" + "─"*60)
    print("  USING SAMPLE DATA / SAMPLE DATA Use කරන්න")
    print("─"*60)
    
    if not os.path.exists(SAMPLE_CSV):
        print(f"\n  ❌ Sample file not found: {SAMPLE_CSV}")
        print(f"  ❌ Sample file නැත: {SAMPLE_CSV}")
        print("\n  Please make sure batch_data_SAMPLE.csv exists!")
        return
    
    # Backup current data if exists
    if os.path.exists(ACTIVE_CSV):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{BACKUP_DIR}/batch_data_backup_{timestamp}.csv"
        shutil.copy2(ACTIVE_CSV, backup_name)
        print(f"\n  💾 Backed up current data to: {backup_name}")
        print(f"  💾 Current data backup කළා: {backup_name}")
    
    # Copy sample to active
    shutil.copy2(SAMPLE_CSV, ACTIVE_CSV)
    
    print(f"\n  ✅ Now using SAMPLE data!")
    print(f"  ✅ දැන් SAMPLE data use කරයි!")
    print(f"\n  The file '{ACTIVE_CSV}' now contains sample data.")
    print(f"  File '{ACTIVE_CSV}' sample data ඇත.")
    print(f"\n  You can now run:")
    print(f"    python train_oil_model.py")


def use_real_data():
    """Switch to real data"""
    print("\n" + "─"*60)
    print("  USING REAL DATA / REAL DATA Use කරන්න")
    print("─"*60)
    
    if not os.path.exists(REAL_CSV):
        print(f"\n  ⚠️  Real data file not found: {REAL_CSV}")
        print(f"  ⚠️  Real data file නැත: {REAL_CSV}")
        print(f"\n  Please create '{REAL_CSV}' with your actual measurements first!")
        print(f"\n  Format should be:")
        print(f"  batch_id,batch_name,drying_days,whole_batch_weight_kg,oil_extracted_ml,notes")
        print(f"  1,My_Batch_01,7,12.5,6800,First real batch")
        print(f"  2,My_Batch_02,7,15.2,8100,Second real batch")
        print(f"  ...")
        return
    
    # Backup current data if exists
    if os.path.exists(ACTIVE_CSV):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{BACKUP_DIR}/batch_data_backup_{timestamp}.csv"
        shutil.copy2(ACTIVE_CSV, backup_name)
        print(f"\n  💾 Backed up old data to: {backup_name}")
        print(f"  💾 පැරණි data backup කළා: {backup_name}")
    
    # Copy real to active
    shutil.copy2(REAL_CSV, ACTIVE_CSV)
    
    # Show the data
    df = pd.read_csv(ACTIVE_CSV)
    print(f"\n  ✅ Now using REAL data!")
    print(f"  ✅ දැන් REAL data use කරයි!")
    print(f"\n  Loaded {len(df)} batches from '{REAL_CSV}':")
    print(df.to_string(index=False))
    
    print(f"\n  Old model files will be replaced when you retrain.")
    print(f"  Retrain කළ විට old model files replace වේවි.")
    print(f"\n  To retrain with real data, run:")
    print(f"  Real data train කිරීමට run කරන්න:")
    print(f"    python train_oil_model.py")


def backup_data():
    """Backup current active data"""
    print("\n" + "─"*60)
    print("  BACKUP DATA / Data Backup කරන්න")
    print("─"*60)
    
    if not os.path.exists(ACTIVE_CSV):
        print("\n  ⚠️  No active data to backup!")
        print("  ⚠️  Backup කිරීමට data නැත!")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"{BACKUP_DIR}/batch_data_backup_{timestamp}.csv"
    shutil.copy2(ACTIVE_CSV, backup_name)
    
    print(f"\n  ✅ Backup saved to: {backup_name}")
    print(f"  ✅ Backup save කළා: {backup_name}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    while True:
        show_menu()
        choice = input("Enter your choice (1-5) / ඔබේ choice enter (1-5): ").strip()
        
        if choice == '1':
            use_sample_data()
        elif choice == '2':
            use_real_data()
        elif choice == '3':
            show_current_data()
        elif choice == '4':
            backup_data()
        elif choice == '5':
            print("\n  Goodbye! / Goodbye!")
            break
        else:
            print("\n  ❌ Invalid choice! Enter 1-5 / වැරදි choice! 1-5 enter කරන්න")
        
        input("\nPress Enter to continue / Continue කිරීමට Enter press...")


if __name__ == '__main__':
    main()