namespace Ametrin.Numerics;

public interface ITensorLike<TSelf> where TSelf : ITensorLike<TSelf>
{
    public int FlatCount { get; }
    public Span<Weight> AsSpan();
    public static abstract TSelf Empty { get; }
    public static abstract TSelf OfSize(TSelf template);
    public static abstract TSelf OfSize(TSelf template, StorageHandle handle);
}
