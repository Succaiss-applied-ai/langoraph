package llm_test

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/zhaocun/langoraph/llm"
)

// ---------------------------------------------------------------------------
// TestLLM_SingleCallLatency
// 测量单次 LLM 调用的实际耗时，确认连通性。
// ---------------------------------------------------------------------------

func TestLLM_SingleCallLatency(t *testing.T) {
	client := newTestClient(t)

	start := time.Now()
	resp, err := client.Chat(context.Background(), []llm.Message{
		{Role: "user", Content: "只回答数字：1+1=？"},
	})
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("LLM call FAILED after %v: %v", elapsed, err)
	}
	t.Logf("✅ Single call latency: %v  content=%q  tokens(in=%d out=%d)",
		elapsed, resp.Content, resp.InputTokens, resp.OutputTokens)
}

// ---------------------------------------------------------------------------
// TestLLM_LongPromptLatency
// 用接近真实场景的长提示词（模拟 NodeKnowledgeCompiler）测量耗时。
// 这个节点最容易超时，因为它要处理大量检索文本。
// ---------------------------------------------------------------------------

func TestLLM_LongPromptLatency(t *testing.T) {
	client := newTestClient(t)

	// 模拟真实的 knowledge_compiler 提示词长度（~3000字）
	fakeChunks := ""
	for i := 0; i < 20; i++ {
		fakeChunks += fmt.Sprintf("[%d] 操作系统是管理计算机硬件与软件资源的计算机程序。操作系统负责管理与配置内存、决定系统资源供需的优先次序、控制输入设备与输出设备、操作网络与管理文件系统等基本任务。操作系统的种类相当多，各种设备安装的操作系统可从简单到复杂，可从移动电话的嵌入式系统到超级计算机的大型操作系统。\n\n", i+1)
	}

	prompt := "你是知识体系编纂器。请把以下检索片段编纂成可命题的知识体系，输出中文：\n\n" + fakeChunks + "\n请输出完整知识体系："

	t.Logf("prompt length: %d chars", len(prompt))

	timeout60s, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	start := time.Now()
	resp, err := client.Chat(timeout60s, []llm.Message{{Role: "user", Content: prompt}})
	elapsed := time.Since(start)

	if err != nil {
		t.Errorf("❌ Long prompt call FAILED after %v: %v", elapsed, err)
		t.Log("这说明 NodeKnowledgeCompiler 节点在 60s 超时下会失败，需要增大超时时间")
		return
	}
	t.Logf("✅ Long prompt latency: %v  output_tokens=%d", elapsed, resp.OutputTokens)
	if elapsed > 50*time.Second {
		t.Logf("⚠️  耗时接近 60s 超时边界，生产环境中该节点有超时风险，建议调大 LLMTimeoutSecs")
	}
}

// ---------------------------------------------------------------------------
// TestLLM_7ConcurrentCalls
// 模拟 7 个 question group 并发调用 LLM 的场景。
// 对比：并发总耗时 vs 串行总耗时，验证并发是否真的生效。
// ---------------------------------------------------------------------------

func TestLLM_7ConcurrentCalls(t *testing.T) {
	client := newTestClient(t)
	const n = 7

	prompt := `请用一句话描述以下知识点，只输出 JSON：{"summary":"..."}\n知识点：操作系统进程调度`

	// --- 串行基准 ---
	serialStart := time.Now()
	for i := 0; i < n; i++ {
		_, err := client.Chat(context.Background(), []llm.Message{{Role: "user", Content: prompt}})
		if err != nil {
			t.Fatalf("serial call %d failed: %v", i, err)
		}
	}
	serialElapsed := time.Since(serialStart)
	t.Logf("串行 %d 次: %v  (avg %.1fs/call)", n, serialElapsed, serialElapsed.Seconds()/n)

	// --- 并发测试 ---
	concurrentStart := time.Now()
	var wg sync.WaitGroup
	errs := make([]error, n)
	latencies := make([]time.Duration, n)

	for i := 0; i < n; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			_, err := client.Chat(context.Background(), []llm.Message{{Role: "user", Content: prompt}})
			latencies[i] = time.Since(start)
			errs[i] = err
		}()
	}
	wg.Wait()
	concurrentElapsed := time.Since(concurrentStart)

	for i, err := range errs {
		if err != nil {
			t.Errorf("concurrent call %d failed: %v", i, err)
		} else {
			t.Logf("  goroutine[%d]: %v", i, latencies[i])
		}
	}
	t.Logf("并发 %d 次: %v", n, concurrentElapsed)
	t.Logf("并发加速比: %.1fx (串行 %.1fs vs 并发 %.1fs)",
		serialElapsed.Seconds()/concurrentElapsed.Seconds(),
		serialElapsed.Seconds(), concurrentElapsed.Seconds())

	if concurrentElapsed > serialElapsed/2 {
		t.Logf("⚠️  并发加速比不足 2x，可能受 API 限速或连接复用问题影响")
	}
}
