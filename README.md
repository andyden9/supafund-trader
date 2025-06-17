# Supafund Trader Agent

An AI-powered prediction system that analyzes startup funding applications on the Supafund platform and generates intelligent predictions about acceptance outcomes. The system leverages exclusive project data, real-time signals, and customizable evaluation criteria to provide data-driven investment insights.

## 🚀 Key Features

### **Comprehensive Data Analysis**
- **Multi-Source Data Integration**: Combines project applications, materials, internal AI analysis reports, and real-time external signals
- **Supabase Integration**: Direct access to Supafund's database containing applications, projects, programs, and evaluation materials
- **Real-Time Signal Enhancement**: Augments static data with live GitHub activity, social sentiment, and market momentum indicators

### **AI-Powered Evaluation Engine**
- **LLM-Driven Predictions**: Uses OpenAI GPT-4 for sophisticated project analysis and outcome prediction
- **Five-Dimensional Assessment**: Evaluates projects across founder/team, market opportunity, technical execution, social/sentiment, and tokenomics
- **Contextual Analysis**: Incorporates program-specific requirements and historical acceptance patterns

### **Customizable Strategy Framework**
- **User-Configurable Weights**: Allows investors to specify evaluation priorities across different dimensions (1-5 scale)
- **Flexible Assessment Levels**: From "Minimal Focus" to "Critical Factor" for each evaluation dimension
- **Adaptive Scoring**: Adjusts prediction models based on user-defined investment philosophy

### **Professional Web Interface**
- **Streamlit-Powered UI**: Clean, responsive interface for application analysis and configuration
- **Real-Time Processing**: Live prediction generation with detailed reasoning and confidence scores
- **Comprehensive Results Display**: Feature breakdowns, LLM prompts, and full state inspection

## 🏗️ Architecture

### **Core Components**

1. **SupafundClient**: Database interface for retrieving applications, projects, programs, and materials
2. **LLMClient**: OpenAI API integration for prediction generation
3. **ExternalAPIClient**: Real-time data fetching from GitHub and other external sources
4. **FeatureAssembler**: Aggregates and scores data across multiple evaluation dimensions
5. **PredictionEngine**: Orchestrates LLM-based analysis and prediction generation
6. **SupafundDataPipeline**: LangGraph-based workflow management system

### **Evaluation Dimensions**

#### **Founder & Team Analysis (Weight: 1-5)**
- Team experience and track record evaluation
- Domain expertise assessment
- Execution capability analysis

#### **Market Opportunity Analysis (Weight: 1-5)**
- Market size and growth potential
- Competitive landscape assessment
- Demand validation metrics

#### **GitHub Technical Analysis (Weight: 1-5)**
- Code quality and development activity
- Technical innovation assessment
- Development momentum tracking

#### **Social & Sentiment Analysis (Weight: 1-5)**
- Community engagement metrics
- Social media presence evaluation
- Market sentiment tracking

#### **Tokenomics Analysis (Weight: 1-5)**
- Token distribution and utility assessment
- Economic model sustainability
- Incentive alignment evaluation

## 🔧 Technical Stack

- **Frontend**: Streamlit (Python web framework)
- **Workflow**: LangGraph (state-based processing pipeline)
- **Database**: Supabase (PostgreSQL with real-time features)
- **AI Engine**: OpenAI GPT-4 (prediction and analysis)
- **External APIs**: GitHub, social media platforms
- **Environment**: Python 3.8+, virtual environment support

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Supabase project credentials
- GitHub token (optional, for enhanced analysis)

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd supafund-trader
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_key
GITHUB_TOKEN=your_github_token  # Optional
```

### 5. Run the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 💡 Usage

1. **Select Application**: Choose a funding application from the dropdown menu
2. **Configure Weights**: Allocate priority levels (1-5) across evaluation dimensions (max 15 total points)
3. **Generate Prediction**: Click "Generate Prediction" to run the analysis pipeline
4. **Review Results**: Examine prediction outcome, confidence score, detailed reasoning, and feature breakdown

## 🔮 How It Works

The system follows a structured pipeline:

1. **Data Retrieval**: Fetches application, project, program data, and submitted materials
2. **Signal Enhancement**: Gathers real-time external signals (GitHub activity, social metrics)
3. **Feature Assembly**: Combines all data sources into a comprehensive feature set
4. **LLM Analysis**: Generates predictions using contextual prompts and user-defined priorities
5. **Results Presentation**: Displays outcomes with detailed reasoning and confidence metrics

## 🎯 Use Cases

- **Investment Decision Support**: Helps VCs and investors evaluate early-stage projects
- **Program Curation**: Assists accelerators in selecting high-potential applications  
- **Risk Assessment**: Provides data-driven insights for funding decisions
- **Market Research**: Analyzes trends and patterns in startup applications

## 🔒 Security & Privacy

- Secure API key management through environment variables
- Read-only database access for analysis
- No sensitive data storage in application state
- Configurable data access controls

## 📈 Future Roadmap

- Integration with additional prediction markets
- Enhanced real-time signal sources
- Advanced ML model training capabilities
- Portfolio optimization features
- Historical performance tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions, issues, or contributions, please open an issue in the GitHub repository or contact the development team.

---

**Note**: This is a prototype system designed for demonstration and research purposes. Always verify predictions with additional due diligence before making investment decisions.