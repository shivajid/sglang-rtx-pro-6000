Below instructions on how to connect Gemini CLI to a model running on sglang. I have built and pushed a custom image of Gemini CLI, to be used as a Coding Harness. It works very well. I have tested with KimiK3 and DeepSeekv4-Flash-0731 model. FBelow are instructions on how to set it up and run it.


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
# Verify Model Access. In this example I have  IP Address 10.0.0.16 for the service/pod, replace with your IP.

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
