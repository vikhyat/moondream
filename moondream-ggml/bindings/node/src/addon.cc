#include <napi.h>
#include "moondream.h"

Napi::Value Add(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 2)
    {
        Napi::TypeError::New(env, "Wrong number of arguments")
            .ThrowAsJavaScriptException();
        return env.Null();
    }

    if (!info[0].IsNumber() || !info[1].IsNumber())
    {
        Napi::TypeError::New(env, "Wrong arguments").ThrowAsJavaScriptException();
        return env.Null();
    }

    double arg0 = info[0].As<Napi::Number>().DoubleValue();
    double arg1 = info[1].As<Napi::Number>().DoubleValue();
    Napi::Number num = Napi::Number::New(env, arg0 + arg1);

    return num;
}

Napi::Value PromptMoondream(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 2)
    {
        Napi::TypeError::New(env, "Wrong number of arguments")
            .ThrowAsJavaScriptException();
        return env.Null();
    }

    if (!info[0].IsString() || !info[1].IsString())
    {
        Napi::TypeError::New(env, "Wrong argument type").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string image_path = info[0].As<Napi::String>().ToString();
    std::string prompt = info[1].As<Napi::String>().ToString();

    int max_gen = 1000;
    bool log_response_stream = true;

    std::string response;
    const bool result = moondream_api_prompt(image_path.c_str(), prompt.c_str(), response, max_gen, log_response_stream);
    printf("result: %d\n", result);
    printf("result: %s\n", response.c_str());
    return Napi::String::New(env, response);
}

Napi::Value InitMoondream(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 2)
    {
        Napi::TypeError::New(env, "Wrong number of arguments")
            .ThrowAsJavaScriptException();
        return env.Null();
    }

    if (!info[0].IsString() || !info[1].IsString())
    {
        Napi::TypeError::New(env, "Wrong argument type").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string text_model = info[0].As<Napi::String>().ToString();
    std::string mm_model = info[1].As<Napi::String>().ToString();
    const bool result = moondream_api_state_init(text_model.c_str(), mm_model.c_str(), 6, true);
    return Napi::Boolean::New(env, result);
}

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    exports.Set(Napi::String::New(env, "add"), Napi::Function::New(env, Add));
    exports.Set(Napi::String::New(env, "init"), Napi::Function::New(env, InitMoondream));
    exports.Set(Napi::String::New(env, "prompt"), Napi::Function::New(env, PromptMoondream));
    return exports;
}

NODE_API_MODULE(addon, Init)