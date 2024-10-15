import unittest
import asyncio
from src.dynamic_batcher import DynamicBatcher

class TestDynamicBatcher(unittest.TestCase):
    def setUp(self):
        self.batcher = DynamicBatcher(max_batch_size=3, max_latency=0.1)

    def test_add_request(self):
        async def test():
            result = await self.batcher.add_request({"input": "test"})
            self.assertEqual(result, [])
            self.assertEqual(len(self.batcher.current_batch), 1)

        asyncio.run(test())

    def test_process_batch(self):
        async def test():
            await self.batcher.add_request({"input": "test1"})
            await self.batcher.add_request({"input": "test2"})
            await self.batcher.add_request({"input": "test3"})
            result = await self.batcher.add_request({"input": "test4"})
            self.assertEqual(len(result), 3)
            self.assertEqual(len(self.batcher.current_batch), 1)

        asyncio.run(test())

    def test_wait_for_batch(self):
        async def test():
            task = asyncio.create_task(self.batcher.add_request({"input": "test"}))
            await asyncio.sleep(0.2)
            self.assertEqual(len(self.batcher.current_batch), 0)
            await task

        asyncio.run(test())

if __name__ == '__main__':
    unittest.main()