import { geolocation, ipAddress } from "@vercel/functions";
import {
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  stepCountIs,
  streamText,
} from "ai";
import { checkBotId } from "botid/server";
import { after } from "next/server";
import { createResumableStreamContext } from "resumable-stream";
import { auth } from "@/app/(auth)/auth";
import { entitlementsByUserType } from "@/lib/ai/entitlements";
import {
  allowedModelIds,
  chatModels,
  DEFAULT_CHAT_MODEL,
  getCapabilities,
} from "@/lib/ai/models";
import { type RequestHints, systemPrompt } from "@/lib/ai/prompts";
import { getLanguageModel } from "@/lib/ai/providers";
import { createDocument } from "@/lib/ai/tools/create-document";
import { editDocument } from "@/lib/ai/tools/edit-document";
import { getWeather } from "@/lib/ai/tools/get-weather";
import { requestSuggestions } from "@/lib/ai/tools/request-suggestions";
import { updateDocument } from "@/lib/ai/tools/update-document";
import { isProductionEnvironment } from "@/lib/constants";
import {
  createStreamId,
  getChatById,
  getMessageCountByUserId,
  getMessagesByChatId,
  saveChat,
  saveMessages,
  updateChatTitleById,
  updateMessage,
} from "@/lib/db/queries";
import type { DBMessage } from "@/lib/db/schema";
import { ChatbotError } from "@/lib/errors";
import { checkIpRateLimit } from "@/lib/ratelimit";
import type { ChatMessage } from "@/lib/types";
import { convertToUIMessages, generateUUID } from "@/lib/utils";
import { generateTitleFromUserMessage } from "../../actions";
import { type PostRequestBody, postRequestBodySchema } from "./schema";

export const maxDuration = 60;

function getStreamContext() {
  try { return createResumableStreamContext({ waitUntil: after }); } catch (_) { return null; }
}

// CORS Headers Helper
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

export async function OPTIONS() {
  return new Response(null, { status: 204, headers: corsHeaders });
}

export async function POST(request: Request) {
  let requestBody: PostRequestBody;
  try {
    const json = await request.json();
    requestBody = postRequestBodySchema.parse(json);
  } catch (_) {
    return new ChatbotError("bad_request:api").toResponse();
  }

  try {
    const { id, message, messages, selectedChatModel, selectedVisibilityType } = requestBody;
    const [, session] = await Promise.all([checkBotId().catch(() => null), auth()]);

    if (!session?.user) return new ChatbotError("unauthorized:chat").toResponse();

    const chatModel = allowedModelIds.has(selectedChatModel) ? selectedChatModel : DEFAULT_CHAT_MODEL;
    await checkIpRateLimit(ipAddress(request));

    const messageCount = await getMessageCountByUserId({ id: session.user.id, differenceInHours: 1 });
    if (messageCount > entitlementsByUserType[session.user.type].maxMessagesPerHour) {
      return new ChatbotError("rate_limit:chat").toResponse();
    }

    const chat = await getChatById({ id });
    let messagesFromDb: DBMessage[] = [];
    let titlePromise: Promise<string> | null = null;

    if (chat) {
      if (chat.userId !== session.user.id) return new ChatbotError("forbidden:chat").toResponse();
      messagesFromDb = await getMessagesByChatId({ id });
    } else if (message?.role === "user") {
      await saveChat({ id, userId: session.user.id, title: "New chat", visibility: selectedVisibilityType });
      titlePromise = generateTitleFromUserMessage({ message });
    }

    const uiMessages = [...convertToUIMessages(messagesFromDb), message as ChatMessage];
    const { longitude, latitude, city, country } = geolocation(request);
    const requestHints: RequestHints = { longitude, latitude, city, country };

    if (message?.role === "user") {
      await saveMessages({
        messages: [{ chatId: id, id: message.id, role: "user", parts: message.parts, attachments: [], createdAt: new Date() }],
      });
    }

    const modelConfig = chatModels.find((m) => m.id === chatModel);
    const modelCapabilities = await getCapabilities();
    const capabilities = modelCapabilities[chatModel];
    const isReasoningModel = capabilities?.reasoning === true;
    const supportsTools = capabilities?.tools === true;
    const modelMessages = await convertToModelMessages(uiMessages);

    const stream = createUIMessageStream({
      originalMessages: undefined,
      execute: async ({ writer: dataStream }) => {
        const result = streamText({
          model: getLanguageModel(chatModel),
          system: systemPrompt({ requestHints, supportsTools }),
          messages: modelMessages,
          tools: {
            getWeather,
            createDocument: createDocument({ session, dataStream, modelId: chatModel }),
            editDocument: editDocument({ dataStream, session }),
            updateDocument: updateDocument({ session, dataStream, modelId: chatModel }),
            requestSuggestions: requestSuggestions({ session, dataStream, modelId: chatModel }),
          },
        });

        dataStream.merge(result.toUIMessageStream({ sendReasoning: isReasoningModel }));
        if (titlePromise) {
          const title = await titlePromise;
          dataStream.write({ type: "data-chat-title", data: title });
          updateChatTitleById({ chatId: id, title });
        }
      },
      generateId: generateUUID,
      onFinish: async ({ messages: finishedMessages }) => {
        if (finishedMessages.length > 0) {
          await saveMessages({
            messages: finishedMessages.map((msg) => ({
              id: msg.id, role: msg.role, parts: msg.parts, createdAt: new Date(), attachments: [], chatId: id,
            })),
          });
        }
      },
    });

    const response = createUIMessageStreamResponse({
      stream,
      async consumeSseStream({ stream: sseStream }) {
        if (!process.env.REDIS_URL) return;
        const streamContext = getStreamContext();
        if (streamContext) {
          const streamId = generateUUID();
          await createStreamId({ streamId, chatId: id });
          await streamContext.createNewResumableStream(streamId, () => sseStream);
        }
      },
    });

    // Apply CORS to the response
    Object.entries(corsHeaders).forEach(([k, v]) => response.headers.set(k, v));
    return response;

  } catch (error) {
    return new ChatbotError("offline:chat").toResponse();
  }
}
