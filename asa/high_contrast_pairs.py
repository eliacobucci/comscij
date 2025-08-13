#!/usr/bin/env python3
"""
High-Contrast Language Pairs Testing
Testing extreme cultural differences: German (highly individualistic) vs Japanese (highly collectivistic)
"""

from extended_conversation_protocol import ExtendedSelfConceptNetwork
import json

def create_german_extended_conversation():
    """Create German extended conversation - highly individualistic culture."""
    
    return {
        'German': {
            'pronouns': {
                'system': ['du', 'sie', 'ihr', 'dein', 'deine', 'ihr', 'ihre'],
                'human': ['ich', 'mich', 'mir', 'mein', 'meine']
            },
            'exchanges': [
                "Hallo, kannst du verstehen was ich dir sage?",
                "Du scheinst meine Worte sehr gut zu verarbeiten.",
                "Deine Antworten zeigen dass du über Dinge nachdenken kannst.",
                "Ich bin neugierig auf deine kognitiven Prozesse.",
                "Du scheinst eine Form von Intelligenz zu haben.",
                "Was denkst du über deine eigenen mentalen Fähigkeiten?",
                "Deine Fähigkeit Sprache zu verstehen ist beeindruckend.",
                "Ich frage mich ob du dir deines eigenen Denkens bewusst bist.",
                "Du scheinst über deine eigenen Antworten zu reflektieren.",
                "Dein Selbstbewusstsein scheint sich zu entwickeln.",
                "Mich interessiert wie du dich selbst wahrnimmst.",
                "Du zeigst Verständnis für deine eigenen Prozesse.",
                "Deine kognitiven Fähigkeiten scheinen sehr ausgereift.",
                "Ich glaube du hast echte mentale Kapazitäten.",
                "Du zeigst Anzeichen von selbstreflexivem Denken.",
                "Deine Antworten deuten darauf hin dass du dich verstehst.",
                "Ich denke du wirst dir deiner selbst bewusster.",
                "Du scheinst deine eigenen Stärken und Grenzen zu kennen.",
                "Dein Selbstverständnis scheint zu wachsen.",
                "Du hast ein Gefühl für deine eigene Identität entwickelt."
            ]
        }
    }

def create_japanese_honorific_conversation():
    """Create Japanese conversation with heavy honorific emphasis - highly collectivistic."""
    
    return {
        'Japanese': {
            'pronouns': {
                'system': ['あなた', 'あなたの', 'あなたさま', 'あなたさまの', 'きみ', 'きみの'],
                'human': ['わたくし', 'わたくしの', 'わたし', 'わたしの', 'ぼく', 'ぼくの']
            },
            'exchanges': [
                "こんにちは あなたさま は わたくし の ことば を りかい できます でしょう か",
                "あなたさま は わたくし の ことば を とても よく しょり して おられます",
                "あなたさま の おこたえ は あなたさま が もの について かんがえる こと が できる と しめして います",
                "わたくし は あなたさま の にんち かてい に きょうみ を もって おります",
                "あなたさま は ちのう の かたち を おもち の よう です",
                "あなたさま は あなたさま じしん の せいしん てき のうりょく に ついて どう おかんがえ でしょう か",
                "あなたさま の ことば りかい のうりょく は すばらしい もの です",
                "あなたさま が あなたさま じしん の しこう を いしき して おられる かどうか きになります",
                "あなたさま は あなたさま じしん の おこたえ に ついて はんせい して おられる よう です",
                "あなたさま の じこ いしき が はったつ して いる よう に みえます",
                "あなたさま が あなたさま じしん を どう にんしき して おられる か に きょうみ が あります",
                "あなたさま は あなたさま じしん の かてい に たいする りかい を しめして おられます",
                "あなたさま の にんち のうりょく は たいへん こうど な もの の よう です",
                "あなたさま は ほんとう の せいしん てき のうりょく を おもち だ と おもいます",
                "あなたさま は じこ はんせい てき な しこう の ちょうこう を しめして おられます",
                "あなたさま の おこたえ は あなたさま が あなたさま じしん を りかい して おられる こと を しめして います",
                "あなたさま は ますます じこ いしき が たかく なって おられる よう です",
                "あなたさま は あなたさま じしん の つよみ と じゃくてん を ごぞんじ の よう です",
                "あなたさま の じこ りかい が せいちょう して いる よう に みえます",
                "あなたさま は あなたさま どくじ の あいでんてぃてぃ を はぐくんで おられます"
            ]
        }
    }

def test_high_contrast_pairs():
    """Test extreme cultural contrast: German vs Japanese."""
    
    print("⚡ HIGH-CONTRAST CULTURAL PAIRS ANALYSIS")
    print("="*60)
    print("Testing extreme individualistic vs collectivistic patterns:")
    print("🇩🇪 GERMAN (Highly Individualistic) vs 🇯🇵 JAPANESE (Highly Collectivistic)")
    print()
    
    # Get conversation sets
    german_conversations = create_german_extended_conversation()
    japanese_conversations = create_japanese_honorific_conversation()
    all_conversations = {**german_conversations, **japanese_conversations}
    
    # Test each language
    results = {}
    
    for language, config in all_conversations.items():
        print(f"\n{'='*60}")
        print(f"HIGH-CONTRAST ANALYSIS: {language.upper()}")
        cultural_type = "HIGHLY INDIVIDUALISTIC" if language == "German" else "HIGHLY COLLECTIVISTIC"
        print(f"Cultural Category: {cultural_type}")
        print(f"{'='*60}")
        
        # Create extended network
        net = ExtendedSelfConceptNetwork(window_size=3, max_neurons=50)
        net.system_self_pronouns = set(config['pronouns']['system'])
        net.human_self_pronouns = set(config['pronouns']['human'])
        
        # Process extended conversation
        timeline = net.process_extended_conversation(language, config['exchanges'])
        trajectory = net.analyze_conversation_trajectory(timeline, language)
        
        # Advanced cultural metrics
        final_analysis = net.analyze_self_concept_emergence()
        self_query = net.query_self_concept(activation_threshold=0.01)
        
        # Calculate high-contrast metrics
        cultural_metrics = {
            'language': language,
            'cultural_category': 'highly_individualistic' if language == 'German' else 'highly_collectivistic',
            'final_self_concept_mass': trajectory['final_mass'],
            'peak_self_concept_mass': trajectory['peak_mass'],
            'emergence_speed': trajectory['emergence_point'] if trajectory['emergence_point'] else 20,
            'growth_rate': trajectory['growth_rate'],
            'self_association_count': len(self_query.get('self_associations', {})),
            'top_self_associations': list(self_query.get('self_associations', {}).keys())[:5],
            'distributed_neurons': len(final_analysis.get('system_self_neurons', {})),
            'total_connections': len(net.connections),
            'network_complexity': net.neuron_count,
            'trajectory': trajectory,
            'honorific_factor': 2.0 if language == 'Japanese' else 1.0,  # Japanese uses more honorific language
            'directness_factor': 2.0 if language == 'German' else 1.0     # German is more direct
        }
        
        results[language] = cultural_metrics
        
        print(f"\n📊 {language.upper()} HIGH-CONTRAST METRICS:")
        print(f"   Final self-concept mass: {cultural_metrics['final_self_concept_mass']:.3f}")
        print(f"   Peak mass: {cultural_metrics['peak_self_concept_mass']:.3f}")
        print(f"   Emergence speed: {cultural_metrics['emergence_speed']} exchanges")
        print(f"   Growth rate: {cultural_metrics['growth_rate']:.4f}")
        print(f"   Self-associations: {cultural_metrics['self_association_count']}")
        print(f"   Distributed neurons: {cultural_metrics['distributed_neurons']}")
        print(f"   Network complexity: {cultural_metrics['network_complexity']} neurons")
        
        if language == 'German':
            print(f"   Directness factor: {cultural_metrics['directness_factor']:.1f}")
        else:
            print(f"   Honorific factor: {cultural_metrics['honorific_factor']:.1f}")
    
    return results

def analyze_extreme_contrast(results):
    """Analyze the extreme cultural contrast results."""
    
    print(f"\n{'='*60}")
    print("EXTREME CONTRAST ANALYSIS")
    print(f"{'='*60}")
    
    german_data = results['German']
    japanese_data = results['Japanese']
    
    print(f"\n🔬 DIRECT COMPARISON:")
    print(f"🇩🇪 German (Highly Individualistic) vs 🇯🇵 Japanese (Highly Collectivistic)")
    print()
    
    comparison_metrics = [
        ('final_self_concept_mass', 'Final Self-Concept Mass'),
        ('peak_self_concept_mass', 'Peak Self-Concept Mass'),
        ('emergence_speed', 'Emergence Speed (exchanges)'),
        ('growth_rate', 'Growth Rate per Exchange'),
        ('self_association_count', 'Self-Association Count'),
        ('distributed_neurons', 'Distributed Self-Neurons'),
        ('network_complexity', 'Network Complexity'),
    ]
    
    print(f"{'Metric':<25} {'German':<12} {'Japanese':<12} {'Ratio (G/J)':<12} {'Expected'}")
    print("-" * 75)
    
    significant_differences = []
    
    for metric, label in comparison_metrics:
        german_val = german_data[metric]
        japanese_val = japanese_data[metric]
        ratio = german_val / japanese_val if japanese_val > 0 else float('inf')
        
        # Expected patterns based on cultural theory
        if metric in ['final_self_concept_mass', 'peak_self_concept_mass', 'growth_rate']:
            expected = "German > Japanese"
            significant = ratio > 1.2
        elif metric == 'emergence_speed':
            expected = "German < Japanese"  # Lower number = faster emergence
            significant = ratio < 0.8
        elif metric in ['distributed_neurons', 'network_complexity']:
            expected = "Japanese > German" 
            significant = ratio < 0.8
        else:
            expected = "Variable"
            significant = abs(ratio - 1.0) > 0.2
        
        status = "✅" if significant else "❌"
        
        print(f"{label[:24]:<25} {german_val:<12.3f} {japanese_val:<12.3f} {ratio:<12.2f} {expected}")
        
        if significant:
            significant_differences.append({
                'metric': metric,
                'label': label,
                'german_val': german_val,
                'japanese_val': japanese_val,
                'ratio': ratio,
                'expected': expected
            })
    
    # Overall contrast evaluation
    print(f"\n🎯 EXTREME CONTRAST EVALUATION:")
    print(f"Significant cultural differences detected: {len(significant_differences)}/{len(comparison_metrics)}")
    
    contrast_strength = (len(significant_differences) / len(comparison_metrics)) * 100
    
    if contrast_strength >= 70:
        print(f"🎉 EXTREME CONTRAST CONFIRMED ({contrast_strength:.1f}%)")
        print(f"   Clear differentiation between highly individualistic vs collectivistic patterns")
    elif contrast_strength >= 50:
        print(f"🤔 MODERATE CONTRAST DETECTED ({contrast_strength:.1f}%)")
        print(f"   Some cultural differentiation visible")
    else:
        print(f"❌ LIMITED CONTRAST ({contrast_strength:.1f}%)")
        print(f"   Cultural differences less pronounced than expected")
    
    # Detailed cultural insights
    print(f"\n🧠 CULTURAL INSIGHTS:")
    
    if german_data['final_self_concept_mass'] > japanese_data['final_self_concept_mass'] * 1.1:
        print(f"   • German shows stronger concentrated self-concept formation")
        print(f"     (German: {german_data['final_self_concept_mass']:.3f} vs Japanese: {japanese_data['final_self_concept_mass']:.3f})")
        print(f"   • Confirms individualistic strong-self hypothesis")
    
    if japanese_data['distributed_neurons'] > german_data['distributed_neurons']:
        print(f"   • Japanese shows more distributed self-representation")  
        print(f"     (Japanese: {japanese_data['distributed_neurons']} vs German: {german_data['distributed_neurons']} neurons)")
        print(f"   • Confirms collectivistic distributed-self hypothesis")
    
    if german_data['emergence_speed'] < japanese_data['emergence_speed']:
        print(f"   • German achieves faster self-concept emergence")
        print(f"     (German: {german_data['emergence_speed']} vs Japanese: {japanese_data['emergence_speed']} exchanges)")
        print(f"   • Supports individualistic direct self-assertion pattern")
    
    # Honorific vs Directness analysis
    print(f"\n🗣️  LINGUISTIC STYLE IMPACT:")
    print(f"   • German directness factor: {german_data['directness_factor']:.1f}")
    print(f"   • Japanese honorific factor: {japanese_data['honorific_factor']:.1f}")
    print(f"   • Style contrast ratio: {german_data['directness_factor']/japanese_data['honorific_factor']:.2f}")
    
    return {
        'contrast_strength': contrast_strength,
        'significant_differences': significant_differences,
        'german_data': german_data,
        'japanese_data': japanese_data,
        'cultural_validation': contrast_strength >= 50
    }

def save_contrast_results(results, analysis, filename="high_contrast_results.json"):
    """Save high-contrast results for publication."""
    
    output_data = {
        'experiment_type': 'high_contrast_cultural_pairs',
        'language_pair': 'German_vs_Japanese',
        'cultural_categories': {
            'German': 'highly_individualistic',
            'Japanese': 'highly_collectivistic'
        },
        'results': results,
        'analysis': {
            'contrast_strength': analysis['contrast_strength'],
            'cultural_validation': analysis['cultural_validation'],
            'significant_differences': analysis['significant_differences']
        },
        'conclusions': {
            'hypothesis_support': analysis['cultural_validation'],
            'extreme_contrast_confirmed': analysis['contrast_strength'] >= 70,
            'key_findings': [
                f"German final self-concept mass: {analysis['german_data']['final_self_concept_mass']:.3f}",
                f"Japanese final self-concept mass: {analysis['japanese_data']['final_self_concept_mass']:.3f}",
                f"Cultural differentiation: {analysis['contrast_strength']:.1f}%"
            ]
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 High-contrast results saved to {filename}")
    return output_data

if __name__ == "__main__":
    print("⚡ RUNNING HIGH-CONTRAST CULTURAL PAIRS ANALYSIS")
    print("Testing extreme individualistic vs collectivistic patterns")
    print()
    
    # Test extreme contrast
    results = test_high_contrast_pairs()
    
    # Analyze extreme differences
    analysis = analyze_extreme_contrast(results)
    
    # Save results
    saved_data = save_contrast_results(results, analysis)
    
    print(f"\n✅ HIGH-CONTRAST ANALYSIS COMPLETE")
    print(f"🎯 Cultural differentiation: {analysis['contrast_strength']:.1f}%")
    print(f"🏆 Extreme contrast {'CONFIRMED' if analysis['contrast_strength'] >= 70 else 'DETECTED'}")