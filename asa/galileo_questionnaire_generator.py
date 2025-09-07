#!/usr/bin/env python3
"""
Galileo Questionnaire Generator
Creates paired comparison questionnaires for multidimensional scaling research

Features:
- Hardwired instructions with customizable reference pair
- User input for concepts/words to compare
- Optimal question display format
- Export for printing or online administration
"""

import streamlit as st
import pandas as pd
from itertools import combinations
import io
from datetime import datetime

def generate_all_pairs(concepts):
    """Generate all unique pairs from a list of concepts"""
    return list(combinations(concepts, 2))

def create_questionnaire_text(concepts, reference_pair, include_numbers=True):
    """Create the complete questionnaire text"""
    
    # Header and instructions
    instructions = f"""GALILEO QUESTIONNAIRE
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

INSTRUCTIONS:
Please estimate how different or "far apart" the following pairs of words are. 
If they are not different at all, say zero (0). If they are different, say a 
larger number. You can say any number you want. To help you know what number 
to write, remember, {reference_pair[0].upper()} and {reference_pair[1].upper()} are 100 units apart.

"""
    
    # Generate all pairs
    pairs = generate_all_pairs(concepts)
    
    # Create questions
    questions = []
    for i, (concept1, concept2) in enumerate(pairs, 1):
        if include_numbers:
            question = f"{i:2d}. {concept1} — {concept2}  _____"
        else:
            question = f"{concept1} — {concept2}  _____"
        questions.append(question)
    
    # Combine all parts
    questionnaire_text = instructions + "\n" + "\n".join(questions)
    questionnaire_text += f"\n\nTotal pairs: {len(pairs)}"
    questionnaire_text += f"\nConcepts: {len(concepts)}"
    
    return questionnaire_text, pairs

def main():
    st.title("🔬 Galileo Questionnaire Generator")
    st.markdown("*Create paired comparison questionnaires for multidimensional scaling research*")
    
    # Sidebar for configuration
    st.sidebar.header("📋 Questionnaire Setup")
    
    # Step 1: Enter concepts
    st.sidebar.subheader("1. Enter Concepts")
    concept_input_method = st.sidebar.radio(
        "How would you like to enter concepts?",
        ["Type each concept", "Paste list (one per line)"]
    )
    
    concepts = []
    if concept_input_method == "Type each concept":
        # Dynamic concept input
        if 'concepts' not in st.session_state:
            st.session_state.concepts = ["concept1", "concept2", "concept3"]
        
        st.sidebar.write("**Current concepts:**")
        for i, concept in enumerate(st.session_state.concepts):
            new_concept = st.sidebar.text_input(f"Concept {i+1}:", value=concept, key=f"concept_{i}")
            st.session_state.concepts[i] = new_concept
        
        # Add/remove concept buttons
        col1, col2 = st.sidebar.columns(2)
        if col1.button("➕ Add Concept"):
            st.session_state.concepts.append(f"concept{len(st.session_state.concepts)+1}")
            st.rerun()
        if col2.button("➖ Remove Last") and len(st.session_state.concepts) > 2:
            st.session_state.concepts.pop()
            st.rerun()
        
        concepts = [c.strip() for c in st.session_state.concepts if c.strip()]
    
    else:
        # Text area input
        concept_text = st.sidebar.text_area(
            "Enter concepts (one per line):",
            value="physics\nchemistry\nbiology\nmathematics\npsychology",
            height=150
        )
        concepts = [c.strip() for c in concept_text.split('\n') if c.strip()]
    
    # Step 2: Choose reference pair
    st.sidebar.subheader("2. Reference Pair")
    if len(concepts) >= 2:
        st.sidebar.write("Choose two concepts for the reference comparison (100 units apart):")
        ref_concept1 = st.sidebar.selectbox("First concept:", concepts, key="ref1")
        ref_concept2 = st.sidebar.selectbox("Second concept:", 
                                          [c for c in concepts if c != ref_concept1], 
                                          key="ref2")
        reference_pair = (ref_concept1, ref_concept2)
    else:
        reference_pair = ("BLANK", "BLANK")
        st.sidebar.warning("Need at least 2 concepts to choose reference pair")
    
    # Step 3: Display options
    st.sidebar.subheader("3. Display Options")
    include_numbers = st.sidebar.checkbox("Include question numbers", value=True)
    show_stats = st.sidebar.checkbox("Show questionnaire statistics", value=True)
    
    # Main content area
    if len(concepts) < 2:
        st.warning("Please enter at least 2 concepts to generate a questionnaire.")
        st.info("A questionnaire needs multiple concepts to create comparison pairs.")
        return
    
    # Generate questionnaire
    questionnaire_text, pairs = create_questionnaire_text(concepts, reference_pair, include_numbers)
    
    # Display statistics
    if show_stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Concepts", len(concepts))
        with col2:
            st.metric("Question Pairs", len(pairs))
        with col3:
            estimated_time = len(pairs) * 0.5  # Rough estimate: 30 seconds per pair
            st.metric("Est. Time (min)", f"{estimated_time:.1f}")
    
    # Display questionnaire preview
    st.subheader("📋 Questionnaire Preview")
    st.text_area("Generated Questionnaire:", questionnaire_text, height=400, disabled=True)
    
    # Export options
    st.subheader("💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download as text file
        filename = f"galileo_questionnaire_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        st.download_button(
            label="📄 Download as Text File",
            data=questionnaire_text,
            file_name=filename,
            mime="text/plain"
        )
    
    with col2:
        # Create CSV for online administration
        df_pairs = pd.DataFrame(pairs, columns=['Concept_1', 'Concept_2'])
        df_pairs['Question_Number'] = range(1, len(pairs) + 1)
        df_pairs['Response'] = ''
        
        csv_buffer = io.StringIO()
        df_pairs.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📊 Download as CSV",
            data=csv_data,
            file_name=f"galileo_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Copy to clipboard (instructions)
        if st.button("📋 Copy Instructions"):
            st.info("Questionnaire text is displayed above - you can select and copy it.")
    
    # Additional information
    st.subheader("ℹ️ About This Questionnaire")
    
    st.info(f"""
    **Reference Pair**: {reference_pair[0]} — {reference_pair[1]} = 100 units
    
    **Instructions for Respondents**:
    - Rate how different each pair of concepts is
    - Use 0 if concepts are identical
    - Use any positive number for different concepts
    - The reference pair ({reference_pair[0]} — {reference_pair[1]}) = 100 units as a comparison guide
    
    **For Researchers**:
    - This generates all unique pairs from your concept list
    - Data can be analyzed using multidimensional scaling (MDS)
    - Consider randomizing question order for different respondents
    - CSV format includes empty Response column for data entry
    """)
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        st.write("**Concept List Management:**")
        if st.button("Clear All Concepts"):
            st.session_state.concepts = ["concept1", "concept2"]
            st.rerun()
        
        st.write("**Question Order:**")
        if st.button("Preview Randomized Order"):
            import random
            shuffled_pairs = pairs.copy()
            random.shuffle(shuffled_pairs)
            st.write("**Randomized question order:**")
            for i, (c1, c2) in enumerate(shuffled_pairs[:10], 1):  # Show first 10
                st.write(f"{i}. {c1} — {c2}")
            if len(shuffled_pairs) > 10:
                st.write(f"... and {len(shuffled_pairs) - 10} more questions")

if __name__ == "__main__":
    main()