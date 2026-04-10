namespace Ametrin.Numerics;

public interface ITensorLike<TSelf> where TSelf : struct, ITensorLike<TSelf>, allows ref struct
{
    public int FlatCount { get; }
    public Span<Weight> AsSpan();

    public static abstract TSelf Empty { get; }
    public static abstract TSelf OfSize(TSelf template);
    public static abstract TSelf OfSize(TSelf template, ArrayHandle handle);
    public static abstract bool HaveSameSize(TSelf a, TSelf b);
}
