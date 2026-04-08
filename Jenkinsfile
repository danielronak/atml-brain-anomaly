pipeline {
    // Run on your local machine
    agent any 

    stages {
        stage('Local Sanity Check (RTX 3050)') {
            steps {
                // Use 'conda run' to cleanly execute without activation issues
                bat '''
                set KMP_DUPLICATE_LIB_OK=TRUE
                conda run -n atml python scripts/train_baseline.py
                '''
            }
        }
        
        stage('Deploy Heavy Training to Azure') {
            steps {
                // If the local test passes, automatically submit the Azure job
                bat '''
                az ml job create -f azure/train_job.yml
                '''
            }
        }
    }
    
    post {
        always {
            echo "Pipeline execution complete."
        }
        success {
            echo "✅ Model passed local tests and is now training on Azure!"
        }
        failure {
            echo "❌ Pipeline failed. Check the logs."
        }
    }
}