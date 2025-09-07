# 📋 Huey Alpha Testing Checklist
## Quick Reference for Galileo Team Testers

### ✅ **Pre-Testing Setup**
- [ ] Huey web interface launches successfully
- [ ] Can access http://localhost:8501
- [ ] Test files available and readable
- [ ] Testing log template ready

### 🧪 **Core Stability Tests (ChatGPT-5 Recommended)**

#### **Same Text, Multiple Runs**
- [ ] Run 1: Process test file → Record eigenvalues & top concepts
- [ ] Run 2: Same file → Compare results 
- [ ] Run 3: Same file → Check consistency
- [ ] **Pass Criteria:** Similar patterns, <10% variance in key metrics

#### **Bootstrapping Test**  
- [ ] Take long conversation, sample 80% randomly
- [ ] Process subset 5 times with different samples
- [ ] Check if core concepts maintain positions
- [ ] **Pass Criteria:** Stable concept neighborhoods

#### **Cross-Session Test**
- [ ] Process text in Session 1
- [ ] Restart Huey completely  
- [ ] Process identical text in Session 2
- [ ] Compare mathematical results
- [ ] **Pass Criteria:** Identical eigenvalue patterns

### 🔍 **Quality Checks**

#### **Function Word Contamination**
- [ ] Review top-weighted concepts
- [ ] **Red Flags:** "something", "the", "and", "very" in top 10
- [ ] **Good:** Meaningful content words dominate

#### **Eigenvalue Sanity**
- [ ] Mix of positive and negative eigenvalues? 
- [ ] High dimensionality preserved (200+ dimensions)?
- [ ] 3D visualization shows appropriate variance %?

#### **Self-Concept Coherence**
- [ ] "myself" associations make theoretical sense?
- [ ] Speaker-concept relationships logical?
- [ ] No obvious nonsense clusters?

### 🚨 **Red Flag Checklist**
Stop testing and report immediately if:
- [ ] Identical values for different metrics
- [ ] Wildly different results from same input
- [ ] System crashes or freezes
- [ ] Negative correlations where positive expected
- [ ] Results contradict basic cognitive theory
- [ ] Memory leaks or severe performance issues

### 📊 **Document These Items**
- [ ] Eigenvalues (first 10)
- [ ] Top concept masses  
- [ ] Key self-concept associations
- [ ] 3D variance percentage
- [ ] Execution time
- [ ] Any error messages
- [ ] Stability across runs (Y/N)
- [ ] Theoretical coherence (Y/N)

### 🎯 **Success Indicators**
- [ ] Consistent results across multiple runs
- [ ] Theoretically coherent concept clusters
- [ ] Meaningful insights not available elsewhere
- [ ] Acceptable performance and reliability
- [ ] Clear separation of content vs function words
- [ ] High-dimensional preservation working properly

### 📧 **Reporting**
- [ ] Weekly alpha report completed
- [ ] Critical issues escalated immediately
- [ ] Testing logs properly documented
- [ ] Confidentiality maintained