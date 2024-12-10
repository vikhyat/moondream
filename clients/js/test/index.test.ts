import axios from 'axios'

import { MoondreamAPI } from "../lib/index"
import { APIConfiguration } from "../lib/types"

jest.mock('axios');

const mockedAxios = axios as jest.Mocked<typeof axios>;

describe("API Interface Tests", () => {

  beforeEach(() => { jest.clearAllMocks});

  it("should return a valid object for a valid query", async () => {
    const mockResponse = { status: 200, data: {type: "vqa", result: "Happy Happy"}}
    mockedAxios.post.mockResolvedValue(mockResponse);

    const config : APIConfiguration = {
      secret: "good-secret",
      endpoint: "endpoint"
    }
    const t = new MoondreamAPI(config);
    const res = await t.request({prompt: 'Prompt', file: "./test/test_img.png"});
    expect(res.status).toBe(200);    
    expect(res.text).toBe("Happy Happy");
  });

  it("should correctly handle auth errors", async() => {
    const mockResponse = { status: 401 }
    mockedAxios.post.mockResolvedValue(mockResponse);

    const config : APIConfiguration = {
      secret: "bad-secret",
      endpoint: "endpoint"
    }
    const t = new MoondreamAPI(config);
    const res = await t.request({prompt: 'Prompt', file: "./test/test_img.png"});
    expect(res.status).toBe(401);    
  });

});
