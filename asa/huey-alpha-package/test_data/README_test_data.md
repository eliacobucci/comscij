# 📁 Test Data for Huey Alpha

This folder contains sample conversation files for testing Huey.

## 📋 **Provided Test Files**

### **chat_transcript_fixed.txt**
- **Type:** ChatGPT conversation about AI consciousness
- **Length:** Medium (~5 pages)
- **Content:** Discussion of human vs AI intelligence, self-concept
- **Use:** Perfect for testing self-concept analysis features
- **Expected:** Rich eigenvalue spectrum, clear concept clusters

### **sample_conversation.txt** 
- **Type:** Human-human dialogue
- **Length:** Short (~2 pages)
- **Content:** General conversation between two people
- **Use:** Testing speaker detection algorithms
- **Expected:** Balanced speaker representation

## 🧪 **Testing Guidelines**

### **Initial Smoke Test**
1. Start with `chat_transcript_fixed.txt`
2. Use default settings (500 neurons, window 7, learning 0.15)
3. Verify all tabs load without errors
4. Check that processing completes in <2 minutes

### **Stability Testing**
1. Process same file multiple times
2. Compare eigenvalue patterns across runs
3. Look for consistent concept clusters
4. Document any variations in results

## 📊 **Expected Results**

### **For chat_transcript_fixed.txt:**
- **Neurons:** ~100-200 concepts
- **Eigenvalues:** Mix of positive/negative (pseudo-Riemannian space)
- **Top concepts:** "myself", "intelligence", "human", "processing"
- **3D variance:** ~15-30% (with appropriate warning)
- **Processing time:** 30 seconds - 2 minutes

### **Success Indicators:**
- ✅ No crashes or error messages
- ✅ Meaningful concept associations
- ✅ Self-concept clusters make sense
- ✅ Eigenvalue spectrum shows complexity
- ✅ Results stable across multiple runs

### **Red Flags:**
- ❌ Identical results every time (too deterministic)
- ❌ Wildly different results every time (unstable)
- ❌ Function words dominate (filtering needed)
- ❌ Processing takes >10 minutes
- ❌ Memory usage grows without bound

## 🔧 **Adding Your Own Test Data**

### **File Format Requirements:**
- **Plain text** (.txt files)
- **UTF-8 encoding** (no special characters that cause issues)
- **Conversation format** with speakers identified
- **Clean text** (no excessive markup or formatting)

### **Speaker Detection Formats:**
Huey can handle various speaker formats:
```
Speaker1: Hello there
Speaker2: Hi back

John: How are you?
Mary: I'm doing well

Person A: What do you think?
Person B: I agree
```

### **Best Practices:**
- **Anonymize** personal information before testing
- **Start small** (1-5 pages) before trying large files
- **Use conversations** rather than monologues
- **Include self-referential content** for best results
- **Document your results** for comparison

## ⚠️ **Privacy & Ethics**

- **Only use appropriate content** for testing
- **Anonymize personal information** in any conversations
- **Do not process confidential materials**
- **Delete test results** containing sensitive information
- **Respect conversation privacy** even in research contexts

---

**Happy Testing! 🧠✨**

Remember: The goal is to stress-test Huey's capabilities and identify any issues before broader release.