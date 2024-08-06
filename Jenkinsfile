def getGitBranchName() { 
                return scm.branches[0].name 
            }
def branchName
def targetBranch

pipeline{
    agent any

    environment {
       DOCKERHUB_USERNAME = "brahim98"
       DEV_TAG = "${DOCKERHUB_USERNAME}/tawasalna-ai:v1.0.0-dev"
       STAGING_TAG = "${DOCKERHUB_USERNAME}/tawasalna-ai:v1.0.0-staging"
       PROD_TAG = "${DOCKERHUB_USERNAME}/tawasalna-ai:v1.0.0-prod"
       JMETER_VERSION = '5.6.3'
               JMETER_HOME = "/path/to/apache-jmeter-${JMETER_VERSION}"
               PERFORMANCE_JMX = 'Performance.jmx'
  }
     parameters {
       string(name: 'BRANCH_NAME', defaultValue: "${scm.branches[0].name}", description: 'Git branch name')
       string(name: 'CHANGE_ID', defaultValue: '', description: 'Git change ID for merge requests')
       string(name: 'CHANGE_TARGET', defaultValue: '', description: 'Git change ID for the target merge requests')
  }
    stages{

      stage('branch name') {
      steps {
        script {
          branchName = params.BRANCH_NAME
          echo "Current branch name: ${branchName}"
        }
      }
    }

    stage('target branch') {
      steps {
        script {
          targetBranch = branchName
          echo "Target branch name: ${targetBranch}"
        }
      }
    }
        stage('Git Checkout'){
            steps{
                git branch: 'Fraud-detection', credentialsId: 'git', url: 'https://github.com/Brahim98-cell/Tawasalna_Mlop_Rating.git'
	    }
        }
        
   /*     stage('Clean Build'){
            steps{
                sh 'rm -rf node_modules'
            }
        }*/

        stage('Install Dependencies') {
            steps {
                // Install required Python packages
                sh 'pip3 install -r requirements.txt'
            }
        }
           /*  stage('Run Script') {
            steps {
                // Run the converted Python script
                sh 'python3 traintest.py'
            }
        }*/
         stage('Run Script') {
            steps {
                // Run the converted Python script and capture the output
                sh 'python3 traintest.py > output.log'
            }
        }
        
  /*   stage('Archive Results') {
            steps {
                // Archive results for later viewing
                archiveArtifacts artifacts: 'output.html', allowEmptyArchive: false
            }
        }

 */
        
        stage('Publish HTML Report') {
            steps {
                publishHTML([allowMissing: false,
                             alwaysLinkToLastBuild: true,
                             keepAll: true,
                             reportDir: '.',
                             reportFiles: 'output.html',
                             reportName: 'HTML Report',
                             reportTitles: 'Product Rating Tawasalna'])
            }
        }
    }
}
    
