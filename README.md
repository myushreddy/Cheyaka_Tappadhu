# ARM Dataset Features

## Feature Types Required

### 1. Static Features from Android Apps

You need to extract two main types of features:

**A. Permissions (from AndroidManifest.xml)**
- Declared in the manifest file
- Hard to obfuscate
- Critical for detecting malware behavior

**B. API Calls (from DEX bytecode)**
- Extracted using static analysis tools (e.g., Androguard)
- Represents app functionality
- Can be paired with permissions for behavioral analysis

---

## Feature Selection Process

### Stage 1: Initial Feature Extraction
- **Total Features**: All possible Android permissions + API calls (~thousands)
- **Filtered to**: **155 features** using Mutual Information (MI)
- **Method**: Select k-best features with highest dependency on malware/benign labels

### Stage 2: GA-RAM Optimization
- **Final Selection**: **48 optimized features**
- **Method**: Genetic algorithm with adaptive mutation
- **Output**: Most robust features against adversarial attacks

---

## Specific Features Identified (Top 48)

### 🔴 Malware-Indicative Features (Positive SHAP scores)

| Category | Feature | SHAP Score | Malware Use Case |
|----------|---------|------------|------------------|
| **SMS Operations** | SEND_SMS | 0.239 | Premium SMS fraud |
| | RECEIVE_SMS | 0.214 | SMS interception |
| | WRITE_SMS | 0.176 | Message manipulation |
| | READ_SMS | 0.192 | Harvesting 2FA codes |
| | sendTextMessage | 0.139 | SMS trojan behavior |
| **Device Info** | READ_PHONE_STATE | 0.201 | Device fingerprinting |
| | getSimSerialNumber | 0.168 | Device tracking |
| | getDeviceId | 0.128 | Unique ID extraction |
| | getSimOperator | 0.131 | Network info collection |
| | getSubscriberId | 0.115 | Subscriber tracking |
| | getLine1Number | 0.012 | Phone number theft |
| **Persistence** | RECEIVE_BOOT_COMPLETED | 0.184 | Auto-start malware |
| | KILL_BACKGROUND_PROCESSES | 0.030 | Disrupting security apps |
| **Network** | CHANGE_WIFI | 0.163 | Network manipulation |
| | ACCESS_NETWORK_STATE | 0.009 | Connectivity checks |
| **Location** | ACCESS_COARSE_LOCATION | 0.150 | Location tracking |
| | ACCESS_FINE_LOCATION | 0.010 | Precise location |
| | ACCESS_LOCATION_EXTRA_COMMANDS | 0.090 | GPS manipulation |
| | getLastKnownLocation | 0.002 | Location retrieval |
| **System Access** | GET_TASKS | 0.142 | App monitoring |
| | SYSTEM_ALERT_WINDOW | 0.121 | Overlay attacks |
| **Contacts** | READ_CONTACTS | 0.109 | Contact theft |
| | WRITE_CONTACTS | 0.021 | Contact injection |
| **Phone Calls** | CALL_PHONE | 0.050 | Unauthorized calls |
| **Data Access** | WRITE_EXTERNAL_STORAGE | 0.040 | File manipulation |
| | READ_EXTERNAL_STORAGE | 0.003 | File access |
| | ContentResolver.query | 0.046 | Database queries |
| **Network I/O** | getInputStream | 0.070 | Data exfiltration |
| | getOutputStream | 0.057 | C&C communication |
| | getNetworkOperator | 0.060 | Network profiling |
| **System Info** | getMessage | 0.102 | Error/message handling |
| | getPackageInfo | 0.014 | App enumeration |
| **Package Management** | DELETE_PACKAGES | 0.016 | App removal |

### 🟢 Benign-Indicative Features (Negative SHAP scores)

| Category | Feature | SHAP Score | Benign Use Case |
|----------|---------|------------|-----------------|
| **Reflection/Loading** | loadClass | -0.265 | Dynamic class loading |
| | getMethod | -0.072 | Reflection APIs |
| **System Testing** | FACTORY_TEST | -0.220 | Manufacturing tests |
| **UI Operations** | requestFocus | -0.213 | User interactions |
| | SET_WALLPAPER | -0.161 | Personalization |
| | EXPAND_STATUS_BAR | -0.158 | UI control |
| **Broadcasting** | BROADCAST_STICKY | -0.175 | System broadcasts |
| **Account Management** | GET_ACCOUNTS | -0.164 | Cloud sync |
| **Media Control** | MODIFY_AUDIO_SETTINGS | -0.152 | Volume/audio |
| **App Management** | getAppPackageName | -0.109 | Package info |
| | INSTALL_SHORTCUT | -0.064 | Home screen shortcuts |
| **User Feedback** | VIBRATE | -0.105 | Haptic feedback |
| **Security** | USE_BIOMETRIC | -0.083 | Fingerprint auth |
| **Browser Data** | WRITE_HISTORY_BOOKMARKS | -0.109 | Browser features |
| **Utilities** | randomUUID | -0.098 | ID generation |

---

## Dataset Requirements

### Training Dataset

**Malware samples: 7,000 apps** (from Drebin, AndroZoo, VirusShare)
- Time period: 2012-2019
- Verified with VirusTotal
- Multiple families: Trojans, Adware, Backdoors, Spyware

**Benign samples: 63,000 apps** (from Google Play Store)
- Categories: Education, Entertainment, Business, Fitness, Games, etc.
- Verified as clean with VirusTotal
- More benign than malware to reduce spatial bias

### Testing Dataset

- **General malware**: 2,500 apps (2020-2022) - temporal bias testing
- **Benign apps**: 2,500 apps (2020-2022)
- **Adversarial samples**: 1,500+ perturbed apps (FGSM, JSMA, GAN, salt-and-pepper, mimicry)
- **Zero-day malware**: 1,500 apps (2023-2024) - newest threats

---

## Feature Extraction Tools Needed

1. **Androguard** - Primary static analysis tool
   - Extracts API calls from DEX bytecode
   - Parses AndroidManifest.xml for permissions

2. **VirusTotal API** - Malware verification
   - Verify malware samples
   - Cross-check benign samples

---

## Feature Format

Each app should be represented as:
- **Binary feature vector**: `[f₁, f₂, ..., f₄₈]`
- **Values**: `{0, 1}` (absence or presence)
- **Label**: `{0, 1}` (benign or malicious)

**Example:**
```
App_ID: com.example.suspicious_app
Features: [1, 0, 1, 1, 0, 0, 1, ...]  (48 values)
Label: 1 (malware)
```

---

## Key Considerations

1. **API-Permission Pairing**: Features work together (e.g., `sendTextMessage` + `SEND_SMS`)
2. **Version Stability**: Permissions less affected by Android version changes
3. **Obfuscation Resistance**: Manifest permissions hard to hide
4. **Behavioral Relevance**: Features must reflect actual malicious operations
5. **Balance**: More benign samples to avoid model bias

---

## Performance Results

This feature set enables:
- **98.6% accuracy** on general malware
- **>90% accuracy** against adversarial attacks!
