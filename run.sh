export PYTHONPATH=$HOME/MADDQN_with_TimesNet/

show_usage() {
    echo "使用方法:"
    echo "  $0 <模式> [選項...]"
    echo ""
    echo "模式:"
    echo "  single                      # 單股票餵入模式"
    echo "  multi                       # 多股票餵入模式"
    echo ""
    echo "選項:"
    echo "  --attention <True|False>    # 是否final agent使用注意力機制 (默認: False)"
    echo "  --reward_shaping <True|False> # 是否使用獎勵塑形 (默認: False)"
    echo ""
    echo "範例:"
    echo "  $0 single --attention True --reward_shaping False"
    echo "  $0 multi --attention True --reward_shaping True --reward_shaping True"
    echo ""
}

if [ $# -eq 0 ]; then
    echo "使用參數："
    echo "  $0 --mode: 運行模式 (single 或 multi)"
    echo "  $0 --attention: 是否使用注意力機制 (True 或 False)"
    echo "  $0 --reward_shaping: 是否使用獎勵塑形 (True 或 False)"
    exit 1
fi
# 檢查是否提供了參數
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# 解析參數
MODE=$1
shift  # 移除第一個參數 (模式)

# 設置默認值
ATTENTION=${ATTENTION:-False}
REWARD_SHAPING=${REWARD_SHAPING:-False}
ADDITIONAL_ARGS=""

# 解析剩餘的命令列參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --attention)
            ATTENTION="$2"
            shift 2
            ;;
        --reward_shaping)
            REWARD_SHAPING="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ 未知參數: $1"
            echo "使用 $0 --help 查看使用說明"
            exit 1
            ;;
    esac
done

# 驗證參數值
if [[ "$ATTENTION" != "True" && "$ATTENTION" != "False" ]]; then
    echo "❌ --attention 參數必須是 True 或 False"
    exit 1
fi

if [[ "$REWARD_SHAPING" != "True" && "$REWARD_SHAPING" != "False" ]]; then
    echo "❌ --reward_shaping 參數必須是 True 或 False"
    exit 1
fi

# 構建完整的命令
BASE_COMMAND="python main.py --attention $ATTENTION --reward_shaping $REWARD_SHAPING $ADDITIONAL_ARGS"

# 根據模式執行相應的命令
case $MODE in
    "single")
        echo "🎯 執行單代理模式..."
        echo "📋 注意力機制: $ATTENTION"
        echo "📋 獎勵塑形: $REWARD_SHAPING"
        $BASE_COMMAND --mode single
        ;;
    "multi")
        echo "🔗 執行多代理模式..."
        echo "📋 注意力機制: $ATTENTION"
        echo "📋 獎勵塑形: $REWARD_SHAPING"
        $BASE_COMMAND --mode multi
        ;;
    "--help"|"-h")
        show_usage
        ;;
    *)
        echo "❌ 未知的運行模式: $MODE"
        echo ""
        echo "支援的模式: train, test, both, analysis, single, multi"
        echo "使用 $0 --help 查看完整使用說明"
        exit 1
        ;;
esac

# 檢查執行結果
if [ $? -eq 0 ]; then
    echo "✅ 執行完成！"
else
    echo "❌ 執行失敗！"
    exit 1
fi