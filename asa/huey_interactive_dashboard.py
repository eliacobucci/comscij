#!/usr/bin/env python3
"""
Huey Interactive Dashboard: Web-based interface for exploring Hebbian concept networks.
Provides real-time visualization and querying capabilities.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import networkx as nx
import numpy as np
import json
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any

class HueyInteractiveDashboard:
    """
    Interactive web dashboard for Huey concept network exploration.
    Provides real-time visualization and query capabilities.
    """
    
    def __init__(self, huey_network=None, query_engine=None):
        """Initialize dashboard with network and query engine."""
        self.network = huey_network
        self.query_engine = query_engine
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
        print("🌐 Huey Interactive Dashboard initialized")
        print("   Features:")
        print("   • 3D concept space visualization")
        print("   • Interactive network graphs")
        print("   • Real-time query interface")
        print("   • Temporal evolution tracking")
        print("   • Speaker comparison tools")
    
    def setup_layout(self):
        """Setup the dashboard layout."""
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🧠 Huey: Hebbian Self-Concept Analysis Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.P("Interactive exploration of concept clusters and self-concept formation",
                      style={'textAlign': 'center', 'color': '#7f8c8d'})
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),
            
            # Main content area
            html.Div([
                # Left panel - Query interface
                html.Div([
                    html.H3("🔍 Query Interface"),
                    
                    # Query type selector
                    html.Label("Query Type:"),
                    dcc.Dropdown(
                        id='query-type-dropdown',
                        options=[
                            {'label': 'Cluster Fellows', 'value': 'cluster_fellows'},
                            {'label': 'Strongest Associations', 'value': 'strongest_associations'},
                            {'label': 'Speaker Differences', 'value': 'speaker_differences'},
                            {'label': 'Temporal Evolution', 'value': 'temporal_evolution'},
                            {'label': 'Network Statistics', 'value': 'network_statistics'}
                        ],
                        value='cluster_fellows'
                    ),
                    
                    # Concept input
                    html.Br(),
                    html.Label("Concept:"),
                    dcc.Input(
                        id='concept-input',
                        type='text',
                        placeholder='Enter concept (e.g., "me", "myself", "identity")',
                        style={'width': '100%'}
                    ),
                    
                    # Speaker selector
                    html.Br(),
                    html.Label("Speaker (optional):"),
                    dcc.Dropdown(
                        id='speaker-dropdown',
                        placeholder="Select speaker or leave blank for all"
                    ),
                    
                    # Threshold slider
                    html.Br(),
                    html.Label("Association Threshold:"),
                    dcc.Slider(
                        id='threshold-slider',
                        min=0.01,
                        max=1.0,
                        step=0.01,
                        value=0.1,
                        marks={0.01: '0.01', 0.1: '0.1', 0.5: '0.5', 1.0: '1.0'}
                    ),
                    
                    # Query button
                    html.Br(),
                    html.Button('Execute Query', id='query-button', 
                               style={'width': '100%', 'padding': '10px'}),
                    
                    # Natural language query
                    html.Br(),
                    html.H4("💬 Natural Language Query"),
                    dcc.Textarea(
                        id='nl-query-input',
                        placeholder='Type natural language query like "show me cluster fellows for \'me\'"',
                        style={'width': '100%', 'height': '100px'}
                    ),
                    html.Button('Ask Huey', id='nl-query-button',
                               style={'width': '100%', 'padding': '10px'})
                    
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 
                         'padding': '20px', 'backgroundColor': '#f8f9fa'}),
                
                # Right panel - Visualizations
                html.Div([
                    # Tabs for different visualizations
                    dcc.Tabs(id='viz-tabs', value='concept-network', children=[
                        dcc.Tab(label='Concept Network', value='concept-network'),
                        dcc.Tab(label='3D Space', value='3d-space'),
                        dcc.Tab(label='Timeline', value='timeline'),
                        dcc.Tab(label='Statistics', value='statistics')
                    ]),
                    
                    # Content area for visualizations
                    html.Div(id='viz-content', style={'padding': '20px'})
                    
                ], style={'width': '75%', 'display': 'inline-block', 'verticalAlign': 'top'})
                
            ], style={'display': 'flex'}),
            
            # Bottom panel - Query results
            html.Div([
                html.H3("📊 Query Results"),
                html.Div(id='query-results', style={'padding': '20px'})
            ], style={'backgroundColor': '#f1f2f6', 'margin': '20px'})
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks for interactivity."""
        
        @self.app.callback(
            Output('query-results', 'children'),
            [Input('query-button', 'n_clicks'),
             Input('nl-query-button', 'n_clicks')],
            [State('query-type-dropdown', 'value'),
             State('concept-input', 'value'),
             State('speaker-dropdown', 'value'),
             State('threshold-slider', 'value'),
             State('nl-query-input', 'value')]
        )
        def execute_query(query_clicks, nl_clicks, query_type, concept, speaker, threshold, nl_query):
            """Execute query and display results."""
            ctx = dash.callback_context
            
            if not ctx.triggered:
                return "No query executed yet."
            
            trigger = ctx.triggered[0]['prop_id']
            
            if not self.query_engine:
                return html.Div([
                    html.H4("❌ No Network Loaded"),
                    html.P("Please load a Huey network to execute queries.")
                ])
            
            try:
                if 'nl-query-button' in trigger and nl_query:
                    # Natural language query
                    result = self.query_engine.natural_language_query(nl_query)
                elif 'query-button' in trigger and concept:
                    # Structured query
                    kwargs = {'concept': concept}
                    if speaker:
                        kwargs['speaker'] = speaker
                    if threshold:
                        kwargs['threshold'] = threshold
                    
                    result = self.query_engine.query(query_type, **kwargs)
                else:
                    return "Please enter a concept or natural language query."
                
                return self._format_query_result(result)
                
            except Exception as e:
                return html.Div([
                    html.H4("❌ Query Error"),
                    html.P(f"Error: {str(e)}")
                ])
        
        @self.app.callback(
            Output('viz-content', 'children'),
            [Input('viz-tabs', 'value')]
        )
        def update_visualization(tab_value):
            """Update visualization based on selected tab."""
            
            if tab_value == 'concept-network':
                return self._create_network_graph()
            elif tab_value == '3d-space':
                return self._create_3d_space()
            elif tab_value == 'timeline':
                return self._create_timeline()
            elif tab_value == 'statistics':
                return self._create_statistics()
            else:
                return html.P("Select a visualization tab")
        
        @self.app.callback(
            Output('speaker-dropdown', 'options'),
            [Input('query-type-dropdown', 'value')]
        )
        def update_speaker_options(query_type):
            """Update speaker dropdown based on available speakers."""
            if self.network and hasattr(self.network, 'speakers'):
                return [{'label': speaker, 'value': speaker} 
                       for speaker in self.network.speakers.keys()]
            return []
    
    def _format_query_result(self, result: Dict[str, Any]) -> html.Div:
        """Format query result for display."""
        
        if 'error' in result:
            return html.Div([
                html.H4("❌ Query Error", style={'color': 'red'}),
                html.P(result['error'])
            ])
        
        # Format based on query type
        query_type = result.get('query', '').split()[0] if 'query' in result else 'unknown'
        
        if 'cluster_fellows' in query_type:
            return self._format_cluster_fellows_result(result)
        elif 'strongest_associations' in query_type:
            return self._format_associations_result(result)
        elif 'speaker_differences' in query_type:
            return self._format_speaker_differences_result(result)
        elif 'network_statistics' in query_type:
            return self._format_statistics_result(result)
        else:
            return html.Pre(json.dumps(result, indent=2))
    
    def _format_cluster_fellows_result(self, result: Dict[str, Any]) -> html.Div:
        """Format cluster fellows query result."""
        
        fellows = result.get('fellow_concepts', [])
        
        if not fellows:
            return html.P(f"No cluster fellows found for '{result.get('target_concept', 'unknown')}'")
        
        # Create table of results
        table_rows = []
        for i, fellow in enumerate(fellows[:20]):  # Limit to top 20
            table_rows.append(html.Tr([
                html.Td(f"{i+1}"),
                html.Td(fellow['concept']),
                html.Td(f"{fellow['strength']:.3f}"),
                html.Td(f"{fellow['mass']:.3f}")
            ]))
        
        return html.Div([
            html.H4(f"🔗 Cluster Fellows for '{result.get('target_concept', 'unknown')}'"),
            html.P(f"Found {result.get('total_fellows', 0)} fellows (showing top 20)"),
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("#"),
                        html.Th("Concept"),
                        html.Th("Strength"),
                        html.Th("Mass")
                    ])
                ]),
                html.Tbody(table_rows)
            ], style={'width': '100%', 'border': '1px solid #ddd'})
        ])
    
    def _format_associations_result(self, result: Dict[str, Any]) -> html.Div:
        """Format strongest associations result."""
        return self._format_cluster_fellows_result(result)  # Same format
    
    def _format_speaker_differences_result(self, result: Dict[str, Any]) -> html.Div:
        """Format speaker differences result."""
        
        analyses = result.get('individual_analyses', {})
        differences = result.get('pairwise_differences', {})
        
        speaker_cards = []
        for speaker, analysis in analyses.items():
            speaker_cards.append(html.Div([
                html.H5(f"🎭 {speaker}"),
                html.P(f"Self-concept mass: {analysis.get('self_concept_mass', 0):.3f}"),
                html.P(f"Blocks processed: {analysis.get('blocks_processed', 0)}")
            ], style={'border': '1px solid #ddd', 'padding': '10px', 'margin': '5px'}))
        
        diff_cards = []
        for comparison, diff in differences.items():
            diff_cards.append(html.Div([
                html.H6(comparison.replace('_vs_', ' vs ')),
                html.P(f"Mass difference: {diff.get('mass_difference', 0):.3f}")
            ], style={'border': '1px solid #eee', 'padding': '5px', 'margin': '2px'}))
        
        return html.Div([
            html.H4("👥 Speaker Self-Concept Analysis"),
            html.Div(speaker_cards),
            html.H5("📊 Pairwise Differences"),
            html.Div(diff_cards)
        ])
    
    def _format_statistics_result(self, result: Dict[str, Any]) -> html.Div:
        """Format network statistics result."""
        
        neuron_stats = result.get('neuron_stats', {})
        connection_stats = result.get('connection_stats', {})
        speaker_stats = result.get('speaker_stats', {})
        
        return html.Div([
            html.H4("📈 Network Statistics"),
            
            html.Div([
                html.H5("🧠 Neurons"),
                html.P(f"Total: {neuron_stats.get('total_neurons', 0)}"),
                html.P(f"Active: {neuron_stats.get('active_neurons', 0)}"),
                html.P(f"Total mass: {neuron_stats.get('total_mass', 0):.3f}"),
                html.P(f"Average mass: {neuron_stats.get('average_mass', 0):.3f}")
            ], style={'display': 'inline-block', 'width': '33%', 'padding': '10px'}),
            
            html.Div([
                html.H5("🔗 Connections"),
                html.P(f"Total: {connection_stats.get('total_connections', 0)}"),
                html.P(f"Strong: {connection_stats.get('strong_connections', 0)}"),
                html.P(f"Average strength: {connection_stats.get('average_strength', 0):.3f}"),
                html.P(f"Max strength: {connection_stats.get('max_strength', 0):.3f}")
            ], style={'display': 'inline-block', 'width': '33%', 'padding': '10px'}),
            
            html.Div([
                html.H5("👥 Speakers"),
                html.Ul([
                    html.Li(f"{speaker}: {stats.get('blocks_processed', 0)} blocks")
                    for speaker, stats in speaker_stats.items()
                ])
            ], style={'display': 'inline-block', 'width': '33%', 'padding': '10px'})
        ])
    
    def _create_network_graph(self):
        """Create interactive network graph visualization."""
        
        if not self.network:
            return html.P("No network loaded")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes (neurons)
        for i, neuron in enumerate(self.network.neurons):
            G.add_node(i, label=neuron.label, mass=neuron.inertial_mass)
        
        # Add edges (connections)
        for i in range(len(self.network.neurons)):
            for j in range(i+1, len(self.network.neurons)):
                strength = self.network.connection_matrix[i][j]
                if strength > 0.05:  # Only show significant connections
                    G.add_edge(i, j, weight=strength)
        
        # Generate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create plotly traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=0.5, color='#888'),
                               hoverinfo='none',
                               mode='lines')
        
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(G.nodes[node]['label'])
            node_size.append(max(10, G.nodes[node]['mass'] * 50))
        
        node_trace = go.Scatter(x=node_x, y=node_y,
                               mode='markers+text',
                               hoverinfo='text',
                               text=node_text,
                               textposition="middle center",
                               marker=dict(size=node_size,
                                         color='lightblue',
                                         line=dict(width=2, color='darkblue')))
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Concept Network Graph',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Concept relationships in the Huey network",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002 ) ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        return dcc.Graph(figure=fig)
    
    def _create_3d_space(self):
        """Create 3D concept space visualization."""
        
        if not self.network:
            return html.P("No network loaded")
        
        # This would use existing 3D visualization logic
        return html.Div([
            html.H4("🌌 3D Concept Space"),
            html.P("3D visualization would be integrated here using existing code"),
            html.P("Shows concept clusters in dimensional space with speaker differentiation")
        ])
    
    def _create_timeline(self):
        """Create temporal evolution visualization."""
        
        if not self.network or not self.network.conversation_history:
            return html.P("No conversation history available")
        
        # Create timeline of concept mass evolution
        steps = []
        masses = []
        
        for i, entry in enumerate(self.network.conversation_history):
            steps.append(i)
            masses.append(entry.get('self_concept_mass', 0))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=masses, mode='lines+markers',
                                name='Self-concept Mass'))
        
        fig.update_layout(title='Self-Concept Evolution Over Time',
                         xaxis_title='Conversation Step',
                         yaxis_title='Self-Concept Mass')
        
        return dcc.Graph(figure=fig)
    
    def _create_statistics(self):
        """Create statistics dashboard."""
        
        if not self.query_engine:
            return html.P("No query engine available")
        
        stats = self.query_engine.query('network_statistics')
        return self._format_statistics_result(stats)
    
    def run(self, host='127.0.0.1', port=8050, debug=True):
        """Run the dashboard server."""
        print(f"\n🚀 Starting Huey Dashboard at http://{host}:{port}")
        print("   Use Ctrl+C to stop the server")
        self.app.run_server(host=host, port=port, debug=debug)

def create_demo_dashboard():
    """Create a demo dashboard with sample data."""
    
    print("🌐 Creating Huey Demo Dashboard...")
    
    # Create dashboard without network (demo mode)
    dashboard = HueyInteractiveDashboard()
    
    return dashboard

if __name__ == "__main__":
    # Run demo dashboard
    demo_dashboard = create_demo_dashboard()
    demo_dashboard.run()