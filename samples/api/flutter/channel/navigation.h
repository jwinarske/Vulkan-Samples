#pragma once

#include "flutter.h"

#include <flutter_embedder.h>

#include "flutter/json_method_codec.h"


namespace flutter
{

struct RouteInformation {
	std::string location;
	std::string state;
	bool replace;
};

static constexpr char kSelectSingleEntryHistory[] = "selectSingleEntryHistory";
static constexpr char kRouteInformationUpdated[] = "routeInformationUpdated";

static void MessageCallback_ChannelNavigation(const FlutterPlatformMessage *message, void *user_data)
{
	auto ctx= reinterpret_cast<flutter *>(user_data)->get_context();
	auto codec = &JsonMethodCodec::GetInstance();
	auto obj = codec->DecodeMethodCall(message->message, message->message_size);
	auto method = obj->method_name();
	auto args = obj->arguments();

	if (method == kSelectSingleEntryHistory) {
		if (args->IsNull()) {
			LOGI("Navigation: Select Single Entry History");
			auto res = codec->EncodeSuccessEnvelope();
			ctx->engine_proc_table_.SendPlatformMessageResponse(ctx->engine_, message->response_handle, res->data(), res->size());
			return;
		}
	}
	else if (method == kRouteInformationUpdated) {
		if (args->HasMember("location") && args->HasMember("state") && args->HasMember("replace")) {
			RouteInformation info{};
			info.location = (*args)["location"].GetString();
			info.state = !(*args)["state"].IsNull() ? (*args)["state"].GetString() : "";
			info.replace = (*args)["replace"].GetBool();
			LOGI("Navigation: Route Information Updated\n\tlocation: {}\n\tstate: {}\n\treplace: {}", "/", "null", "false");
			auto res = codec->EncodeSuccessEnvelope();
			ctx->engine_proc_table_.SendPlatformMessageResponse(ctx->engine_, message->response_handle, res->data(), res->size());
			return;
		}
	}

	LOGI("[Navigation] Unhandled: {}", method);
	ctx->engine_proc_table_.SendPlatformMessageResponse(ctx->engine_, message->response_handle, nullptr, 0);
}

}        // namespace flutter
