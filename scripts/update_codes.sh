# !/bin/bash

echo "--------------------------"
echo "update tsproj codes..."
echo "--------------------------"
git checkout main
echo "Successfully checked out main."

git add .
git commit -m "init project"

git pull
echo "Successfully pulled the latest changes."

git push
echo "Successfully checked out master and updated the code."

# push utils
echo "--------------------------"
echo "update utils codes..."
echo "--------------------------"
cd utils

git add .
git commit -m "update"

git pull
echo "Successfully pulled the latest changes."

git push
echo "Successfully checked out master and updated the code."
