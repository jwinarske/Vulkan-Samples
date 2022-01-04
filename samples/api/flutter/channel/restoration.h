#pragma once

#include "flutter.h"

#include <flutter_embedder.h>

#include "flutter/standard_method_codec.h"

namespace flutter
{

static constexpr char kMethodGet[] = "get";

static void MessageCallback_ChannelRestoration(const FlutterPlatformMessage *message, void *user_data)
{
	auto ctx= reinterpret_cast<flutter *>(user_data)->get_context();
	auto codec = &StandardMethodCodec::GetInstance();
	auto obj = codec->DecodeMethodCall(message->message, message->message_size);
	auto method = obj->method_name();
	auto args = obj->arguments();

	if (method == kMethodGet) {
		auto res = codec->EncodeSuccessEnvelope();
		ctx->engine_proc_table_.SendPlatformMessageResponse(ctx->engine_, message->response_handle, res->data(), res->size());
		return;
	}

	LOGI("[Restoration] Unhandled: {}", method);
	ctx->engine_proc_table_.SendPlatformMessageResponse(ctx->engine_, message->response_handle, nullptr, 0);
}

}        // namespace flutter
