#pragma once

#include "flutter.h"

#include <flutter_embedder.h>

#include "flutter/json_method_codec.h"

namespace flutter
{

struct MethodSetApplicationSwitcherDescription
{
	std::string label;
	uint32_t    primaryColor;
};

static constexpr char kMethodSetApplicationSwitcherDescription[] = "SystemChrome.setApplicationSwitcherDescription";

static constexpr char kMethodSetSystemUiOverlayStyle[]    = "SystemChrome.setSystemUIOverlayStyle";
static constexpr char kMethodSetEnabledSystemUIOverlays[] = "SystemChrome.setEnabledSystemUIOverlays";
static constexpr char kMethodSystemNavigatorPopMethod[]   = "SystemNavigator.pop";

static constexpr char kBadArgumentsError[]           = "Bad Arguments";
static constexpr char kUnknownClipboardFormatError[] = "Unknown Clipboard Format";
static constexpr char kFailedError[]                 = "Failed";
static constexpr char kMethodClipboardHasStrings[]   = "Clipboard.hasStrings";
static constexpr char kMethodClipboardSetData[]      = "Clipboard.setData";
static constexpr char kMethodClipboardGetData[]      = "Clipboard.getData";
static constexpr char kSystemNavigatorPopMethod[]    = "SystemNavigator.pop";

static constexpr char kTextPlainFormat[] = "text/plain";

static constexpr char kPlaySoundMethod[] = "SystemSound.play";
static constexpr char kSoundTypeAlert[]  = "SystemSoundType.alert";
static constexpr char kSoundTypeClick[]  = "SystemSoundType.click";

static void MessageCallback_ChannelPlatform(const FlutterPlatformMessage *message, void *user_data)
{
	auto ctx    = reinterpret_cast<flutter *>(user_data)->get_context();
	auto codec  = &JsonMethodCodec::GetInstance();
	auto obj    = codec->DecodeMethodCall(message->message, message->message_size);
	auto method = obj->method_name();
	auto args   = obj->arguments();

	if (method == kMethodSetApplicationSwitcherDescription)
	{
		if (args->HasMember("label") && args->HasMember("primaryColor"))
		{
			MethodSetApplicationSwitcherDescription description{};
			description.label        = (*args)["label"].GetString();
			description.primaryColor = (*args)["primaryColor"].GetUint();
			LOGI("Platform: ApplicationSwitcherDescription\n\tlabel: \"{}\"\n\tprimaryColor: {}", description.label, description.primaryColor);
			auto res = codec->EncodeSuccessEnvelope();
			ctx->engine_proc_table_.SendPlatformMessageResponse(ctx->engine_, message->response_handle, res->data(), res->size());
			return;
		}
	}
	else if (method == kMethodClipboardHasStrings)
	{
		if (args->IsString())
		{
			auto format = args->GetString();
			if (0 == strcmp(format, kTextPlainFormat))
			{
				rapidjson::Document result;
				result.SetBool(false);
				auto res = codec->EncodeSuccessEnvelope(&result);
				ctx->engine_proc_table_.SendPlatformMessageResponse(ctx->engine_, message->response_handle, res->data(), res->size());
				return;
			}
		}
	}
	else if (method == kMethodClipboardSetData)
	{
		if (args->HasMember("text") && !((*args)["text"].IsNull()))
		{
			LOGD("Clipboard Data Set: \n{}", (*args)["text"].GetString());
		}
		auto res = codec->EncodeSuccessEnvelope();
		ctx->engine_proc_table_.SendPlatformMessageResponse(ctx->engine_, message->response_handle, res->data(),
		                                                    res->size());
		return;
	}

	LOGI("[Platform] Unhandled: {}", method);
	ctx->engine_proc_table_.SendPlatformMessageResponse(ctx->engine_, message->response_handle, nullptr, 0);
}

}        // namespace flutter
