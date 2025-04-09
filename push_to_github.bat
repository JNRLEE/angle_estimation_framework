@echo off
echo Setting up Git repository and pushing to GitHub...

REM Initialize Git repository
git init

REM Add all files to staging
git add .

REM Make initial commit
git commit -m "Initial commit with Docker and PyTorch setup"

REM Add remote repository
git remote add origin https://github.com/JNRLEE/LDV_Reorientation.git

REM Push to main branch
git push -u origin main

echo Done! Check for any errors above.
pause 