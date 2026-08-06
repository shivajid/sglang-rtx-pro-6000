


```
## Update package lists

sudo apt-get update

## Install git, curl, and native compilation tools
sudo apt-get install -y git curl build-essential python3

## Install Node.js 20.x LTS via NodeSource
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

## Verify versions
node -v   # Should be v20.x or higher
npm -v    # Should be 10.x or higher
```
##Verify Model Access
```
# Verify Model Access. here IP Address is the IP of the service/pod. -> 10.0.0.16. Find your IP.

# List the model
curl http://10.0.0.16:30000/v1/models 

# Sample Prompt 

curl http://10.0.0.16:30000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "messages": [
      {"role": "user", "content": "Hello! What model are you?"}
    ],
    "max_tokens": 128,
    "temperature": 0.6
  }
 
```

## Set Environment
```
export SGLANG_BASE_URL=http://10.0.0.16:30000/v1
export GEMINI_MODEL="deepseek-ai/DeepSeek-V4-Flash-0731"
export GEMINI_DEFAULT_AUTH_TYPE="sglang"
```


## 3. Run Gemini CLI
```
npx @shivajidnpm2026/gemini-cli
```
