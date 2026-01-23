# MH-100K Dataset Analysis Report

## Executive Summary

⚠️ **NOT DIRECTLY COMPATIBLE** with your ARM project requirements.  
This dataset requires **significant preprocessing** before use.

---

## Dataset Overview

| Metric | Value | Your Requirement |
|--------|-------|------------------|
| **Total Samples** | 101,934 | 70,000 (training) |
| **Malware Samples** | 12,721 (12.48%) | 7,000 (10%) |
| **Benign Samples** | 89,213 (87.52%) | 63,000 (90%) |
| **Total Features** | ~18,630 features | **48 features** |

✅ **Sample size**: Exceeds requirements  
✅ **Class distribution**: Close to your 10:90 ratio  
❌ **Features**: Wrong set - needs complete re-processing

---

## Critical Issues

### 🔴 Issue #1: Feature Count Mismatch

**Dataset has:** ~18,630 features  
**You need:** 48 optimized features from GA-RAM

This dataset contains:
- All possible Android permissions
- All possible API calls (thousands)
- Metadata columns (SHA256, package name, API levels)

**Your project needs:** Only the 48 specific features identified through:
1. Mutual Information filtering → 155 features
2. GA-RAM optimization → 48 final features

### 🔴 Issue #2: Feature Selection Process Missing

This dataset is **pre-GA-RAM optimization**. It's a raw feature extraction dataset.

**What this means:**
- Features haven't been filtered for adversarial robustness
- No SHAP-based feature importance applied
- Includes many irrelevant/redundant features

### 🟡 Issue #3: Feature Overlap Verification Needed

**Your required features** (from README):
- `SEND_SMS`, `READ_SMS`, `WRITE_SMS`, `READ_PHONE_STATE`
- `sendTextMessage`, `getDeviceId`, `getSimSerialNumber`
- `RECEIVE_BOOT_COMPLETED`, `ACCESS_COARSE_LOCATION`
- etc. (48 total)

**Dataset contains** (confirmed present):
- ✅ `Permission::SEND_SMS` (index 5099)
- ✅ `Permission::READ_SMS` (index 3253)
- ✅ `Permission::READ_PHONE_STATE` (index 2706)
- ✅ `APICall::Landroid/telephony/SmsManager.sendTextMessage()` (index 5425)
- ✅ `APICall::Landroid/telephony/TelephonyManager.getDeviceId()` (index 1768)
- ✅ `Permission::GET_ACCOUNTS` (index 2701)
- ✅ `Permission::VIBRATE` (index 2703)
- ✅ `APICall::Landroid/view/View.requestFocus()` (index 30)

**Good news:** Many of your required features ARE present in this dataset.

**Bad news:** They're buried among 18,000+ other features.

---

## What You Need To Do

### Option 1: Extract Your 48 Features (Recommended)

**Steps:**
1. Map your 48 required features to this dataset's column indices
2. Extract only those 48 columns from `mh100-features-all.csv`
3. Create new dataset with shape: (101,934 samples × 48 features)
4. Verify feature values are binary (0/1)

**Advantages:**
- Uses proven feature set from ARM research
- Maintains adversarial robustness
- Follows your paper's methodology

**Challenge:**
- Need exact feature name mapping
- Some features might have slightly different names (e.g., API call signatures)

### Option 2: Re-run Feature Selection Pipeline

**Steps:**
1. Use all 18,630 features as input
2. Apply Mutual Information filtering → reduce to ~155 features
3. Run GA-RAM algorithm → optimize to 48 features
4. Compare with original ARM paper features

**Advantages:**
- Generates features optimized for THIS specific dataset
- Follows complete ARM methodology

**Challenge:**
- Requires implementing/running GA-RAM algorithm
- Time-consuming (genetic algorithm training)
- Results may differ from paper

### Option 3: Use Dataset As-Is for Baseline

**Steps:**
1. Use all 18,630 features for initial model training
2. Apply dimensionality reduction (PCA, feature selection)
3. Compare performance with ARM's 48-feature approach

**Advantages:**
- Quick start without feature engineering
- Establishes baseline performance

**Challenge:**
- Won't achieve ARM's adversarial robustness
- High computational cost (curse of dimensionality)
- Not following ARM methodology

---

## Feature Name Mapping Examples

Your required features may appear with different naming conventions:

| Your README Feature | MH-100K Dataset Feature | Index |
|---------------------|-------------------------|-------|
| `SEND_SMS` | `Permission::SEND_SMS` | 5099 |
| `READ_SMS` | `Permission::READ_SMS` | 3253 |
| `READ_PHONE_STATE` | `Permission::READ_PHONE_STATE` | 2706 |
| `sendTextMessage` | `APICall::Landroid/telephony/SmsManager.sendTextMessage()` | 5425 |
| `getDeviceId` | `APICall::Landroid/telephony/TelephonyManager.getDeviceId()` | 1768 |
| `requestFocus` | `APICall::Landroid/view/View.requestFocus()` | 30 |
| `GET_ACCOUNTS` | `Permission::GET_ACCOUNTS` | 2701 |
| `VIBRATE` | `Permission::VIBRATE` | 2703 |

**Next step:** Create complete mapping for all 48 features.

---

## Dataset Strengths

✅ **Large scale**: 101K+ samples - excellent for training  
✅ **Good class balance**: 12.5% malware close to your 10% target  
✅ **VirusTotal verification**: Multi-engine malware validation  
✅ **Pre-extracted features**: Saves Androguard processing time  
✅ **Includes your features**: Most/all of your 48 features are present  

---

## Recommendation

### 🎯 Recommended Approach: **Option 1 - Feature Extraction**

**Action Plan:**
1. **Create feature mapping** - Map all 48 required features to MH-100K indices
2. **Extract subset** - Pull only those 48 columns from the dataset
3. **Validate format** - Ensure binary (0/1) encoding
4. **Split data** - 80:20 train/test following ARM methodology
5. **Train model** - Use Random Forest with 100 trees

**I can help you:**
- Create the feature mapping script
- Extract the 48-feature subset
- Validate data format
- Set up the training pipeline

Would you like me to start creating the feature extraction script?

---

## Files in Dataset

- ✅ `mh100-features-all.csv` - Full feature matrix (101,934 × 18,630)
- ✅ `mh100-features-classes.csv` - Feature column names/mapping
- ✅ `mh100_vt-labels.csv` - VirusTotal multi-engine labels (40 engines)
- ✅ `mh100_labels.csv` - Primary binary labels (malware/benign)
- ⚠️ `mh100.npz` - Compressed NumPy format (alternative to CSV)

---

## Next Steps

**Immediate actions:**
1. ✅ Dataset downloaded and analyzed
2. ⏭️ Create feature mapping for all 48 features
3. ⏭️ Extract 48-feature subset
4. ⏭️ Validate against ARM paper requirements
5. ⏭️ Begin model training

**Question for you:**  
Do you want to proceed with **Option 1** (extracting your 48 features), or would you prefer to try **Option 2** (re-running GA-RAM optimization on this dataset)?
