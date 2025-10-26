# Generate SSH for GitHub

*This guide explains how to generate and configure SSH keys for secure GitHub access.*

1. Generate a new SSH key:
```
ssh-keygen -t ed25519 -C "your_email@example.com"
```
For 'Enter file in which to save the key', just press enter without typing anything to use default location.

2. Start ssh-agent and add the key:
```
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```
3. Print the public key:
```
cat ~/.ssh/id_ed25519.pub
```
4. Copy that output → go to GitHub → Settings → SSH and GPG keys → New SSH key, paste it in.
5. Test again:
```
ssh -T git@github.com
```
You will be prompted: Are you sure you want to continue connecting (yes/no/[fingerprint])? 
-> Enter yes

You should see:
```
Hi your-username! You've successfully authenticated, but GitHub does not provide shell access.
```

Then:
- clone project repo
```
git clone git@github.com:ChinSekYi/mlops-fyp.git
```
- Ensure your repo is pointing to SSH:
```
git remote set-url origin git@github.com:ChinSekYi/mlops-fyp.git
```