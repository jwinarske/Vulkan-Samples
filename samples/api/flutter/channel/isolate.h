#pragma once

#include "common/logging.h"
#include "flutter.h"

#include <flutter_embedder.h>

#include "flutter/standard_method_codec.h"

namespace flutter {

static void MessageCallback_ChannelIsolate(const FlutterPlatformMessage *message, void *user_data) {
	auto ctx = reinterpret_cast<flutter *>(user_data)->get_context();

    std::string msg;
	msg.append(reinterpret_cast<const char*>(message->message));
	msg.resize(message->message_size);
	LOGD("Root Isolate Service ID: \"{}\"", msg);

	auto res = StandardMethodCodec::GetInstance().EncodeSuccessEnvelope();
	ctx->engine_proc_table_.SendPlatformMessageResponse(ctx->engine_, message->response_handle, res->data(), res->size());
}

}
