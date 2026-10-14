import pytest
import time
from core.execution_queue import ExecutionQueueManager

def test_execution_queue_fifo_and_deduplication():
    queue = ExecutionQueueManager()
    
    results = []
    
    def task_fn(val):
        time.sleep(0.05)
        results.append(val)
        return val
        
    t1 = queue.submit_task(task_fn=task_fn, task_name="t1", task_type="TASK_1", args=("first",))
    t2 = queue.submit_task(task_fn=task_fn, task_name="t2", task_type="TASK_2", args=("second",))
    # Duplicate of TASK_2
    t3 = queue.submit_task(task_fn=task_fn, task_name="t3", task_type="TASK_2", args=("third",), deduplicate=True)
    
    assert t3.get("status") == "DEDUPLICATED"  # Should be deduplicated
    
    # Wait for completion
    time.sleep(0.4)
    
    assert "first" in results
    assert "second" in results
    assert "third" not in results
    
    status = queue.get_status()
    assert "queued_count" in status
    assert "is_running" in status
