using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;

namespace ray.helper;

public static class MnistLoader
{
    public static (
        List<List<double>> Images,
        List<List<double>> Labels
    ) Load(
        string imageFile,
        string labelFile,
        int maxSamples = int.MaxValue)
    {
        using Stream imageStream = OpenFile(imageFile);
        using Stream labelStream = OpenFile(labelFile);

        using var imageReader = new BinaryReader(imageStream);
        using var labelReader = new BinaryReader(labelStream);

        int imageMagic = ReadBigEndianInt32(imageReader);
        int imageCount = ReadBigEndianInt32(imageReader);
        int rows = ReadBigEndianInt32(imageReader);
        int columns = ReadBigEndianInt32(imageReader);

        int labelMagic = ReadBigEndianInt32(labelReader);
        int labelCount = ReadBigEndianInt32(labelReader);

        if (imageMagic != 2051)
            throw new InvalidDataException(
                $"Ungültige MNIST-Bilddatei. Magic: {imageMagic}");

        if (labelMagic != 2049)
            throw new InvalidDataException(
                $"Ungültige MNIST-Labeldatei. Magic: {labelMagic}");

        if (imageCount != labelCount)
            throw new InvalidDataException(
                "Anzahl Bilder und Labels stimmt nicht überein.");

        int count = Math.Min(
            Math.Min(imageCount, labelCount),
            maxSamples);

        var images = new List<List<double>>(count);
        var labels = new List<List<double>>(count);

        int pixelsPerImage = rows * columns;

        for (int sample = 0; sample < count; sample++)
        {
            var image = new List<double>(pixelsPerImage);

            for (int pixel = 0; pixel < pixelsPerImage; pixel++)
            {
                byte value = imageReader.ReadByte();

                // 0..255 -> 0..1
                image.Add(value / 255.0);
            }

            byte digit = labelReader.ReadByte();

            var target = new List<double>(10);

            for (int i = 0; i < 10; i++)
            {
                target.Add(i == digit ? 1.0 : 0.0);
            }

            images.Add(image);
            labels.Add(target);
        }

        return (images, labels);
    }

    private static Stream OpenFile(string path)
    {
        Stream stream = File.OpenRead(path);

        if (path.EndsWith(
            ".gz",
            StringComparison.OrdinalIgnoreCase))
        {
            return new GZipStream(
                stream,
                CompressionMode.Decompress);
        }

        return stream;
    }

    private static int ReadBigEndianInt32(BinaryReader reader)
    {
        byte[] bytes = reader.ReadBytes(4);

        if (bytes.Length != 4)
            throw new EndOfStreamException();

        if (BitConverter.IsLittleEndian)
            Array.Reverse(bytes);

        return BitConverter.ToInt32(bytes, 0);
    }
}