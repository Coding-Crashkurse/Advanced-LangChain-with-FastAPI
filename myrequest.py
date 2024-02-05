import asyncio

import httpx


async def make_request(endpoint):
    url = f"http://localhost:6677/{endpoint}"
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(url)
        if response.status_code == 200:
            print(f"Response from {endpoint}: {response.json()}")
        else:
            print(
                f"Error from {endpoint}: {response.status_code}, redirected to: {response.url}"
            )


async def main():
    tasks = [make_request("get-all-ids/"), make_request("quick-response/")]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
